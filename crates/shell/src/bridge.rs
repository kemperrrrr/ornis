//! IPC bridge between the embedded page and the engine thread.
//!
//! This mirrors the `/api` semantics of `editor_backend::remote` without
//! touching that transport:
//!
//! * page → engine: `window.ipc.postMessage(JSON)` envelopes shaped exactly
//!   like `POST /api/command` bodies (`{"type", "request_id"?, "data"}`).
//!   [`parse_incoming`] validates the envelope and wraps the command as
//!   [`UiCommand::WithRequestId`](editor_backend::ipc::UiCommand::WithRequestId),
//!   so the engine emits a correlated
//!   [`GameEvent::CommandCompleted`](editor_backend::ipc::GameEvent::CommandCompleted).
//!   Every post gets an explicit acknowledgement ([`AckOutcome`], same JSON
//!   shape as the HTTP `{"accepted", "request_id"[, "error"]}` ack).
//! * engine → page: [`events_message_json`] serializes [`GameEvent`]s with
//!   the same canonical variant shapes as `/api/events` plus a transport
//!   `sequence` sibling; [`dispatch_script`] delivers the payload to the
//!   page as a `CustomEvent("ornis-ipc")`.
//!
//! Everything here is pure functions over channel-free values, so the whole
//! protocol is unit-tested without opening a window. The `wry` wiring
//! (`with_ipc_handler`, `evaluate_script`) lives in the binary.

use editor_backend::ipc::{EventSeq, GameEvent, RequestId, UiCommand};
use editor_backend::remote::parse_command_payload;

use crate::error::BridgeError;

/// A validated page post: the transport id plus the parse outcome.
///
/// The id is assigned before validation (a positive client `request_id` is
/// honoured, otherwise one is allocated), so even rejected posts get a
/// correlated acknowledgement — the same guarantee as the HTTP transport.
#[derive(Debug, Clone)]
pub struct IncomingPost {
    /// Transport id echoed by the ack and the completion event.
    pub request_id: RequestId,
    /// The parsed command, or why the post was rejected.
    pub command: Result<UiCommand, BridgeError>,
}

impl IncomingPost {
    /// Wraps a successfully parsed command as
    /// [`UiCommand::WithRequestId`](editor_backend::ipc::UiCommand::WithRequestId),
    /// ready to send over `cmd_tx`; rejections pass through unchanged.
    pub fn into_queued(self) -> (RequestId, Result<UiCommand, BridgeError>) {
        let request_id = self.request_id;
        match self.command {
            Ok(command) => (
                request_id,
                Ok(UiCommand::WithRequestId {
                    request_id,
                    command: Box::new(command),
                }),
            ),
            Err(error) => (request_id, Err(error)),
        }
    }
}

/// Synchronous acknowledgement for one posted command.
///
/// Same outcome vocabulary as the HTTP transport: `accepted` only means the
/// message was validated and queued — the engine may complete it later with
/// a `CommandCompleted` event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AckOutcome {
    /// The command was validated and queued on the engine thread.
    Accepted {
        /// Transport id of the queued command.
        request_id: RequestId,
    },
    /// The command was rejected before reaching the engine.
    Rejected {
        /// Transport id assigned to the rejected post.
        request_id: RequestId,
        /// Human-readable reason (mirrors the HTTP ack `error` strings).
        error: String,
    },
}

impl AckOutcome {
    /// The transport id of the acknowledged post.
    pub fn request_id(&self) -> RequestId {
        match self {
            AckOutcome::Accepted { request_id } | AckOutcome::Rejected { request_id, .. } => {
                *request_id
            }
        }
    }

    /// Whether the command was queued (`true`) or rejected (`false`).
    pub fn accepted(&self) -> bool {
        matches!(self, AckOutcome::Accepted { .. })
    }

    /// Serializes to the HTTP-ack-compatible JSON shape
    /// (`{"accepted", "request_id"[, "error"]}`).
    pub fn to_json(&self) -> String {
        match self {
            AckOutcome::Accepted { request_id } => ack_json(*request_id, true, None),
            AckOutcome::Rejected { request_id, error } => ack_json(*request_id, false, Some(error)),
        }
    }
}

/// Allocates the next server-side request id, skipping the reserved `0`.
///
/// Mirrors the HTTP allocator: ids are monotonic and saturate instead of
/// wrapping, so a client cursor starting at `0` never collides.
pub fn allocate_request_id(next: &mut RequestId) -> RequestId {
    let request_id = RequestId::new(next.get().max(1));
    *next = request_id.next();
    request_id
}

/// Parses one `window.ipc.postMessage` payload into a post with its id.
///
/// The envelope is byte-compatible with `POST /api/command` bodies:
/// `{"type": …, "request_id"?: u64, "data": …}`. A positive client
/// `request_id` is honoured (and advances the allocator past it, so later
/// generated ids never collide); otherwise an id is allocated from `next`.
///
/// Validation failures land in [`IncomingPost::command`], never in a
/// `Result`: the caller still owes the page an ack carrying the assigned
/// id (see [`AckOutcome::Rejected`]).
pub fn parse_incoming(body: &str, next: &mut RequestId) -> IncomingPost {
    let value: Option<serde_json::Value> = serde_json::from_str(body).ok();
    let request_id = value
        .as_ref()
        .and_then(|envelope| envelope.get("request_id"))
        .and_then(|id| id.as_u64())
        .filter(|&id| id > 0)
        .map(RequestId::new)
        .inspect(|&id| {
            if id >= *next {
                *next = id.next();
            }
        })
        .unwrap_or_else(|| allocate_request_id(next));
    let command = (|| {
        let envelope = value.ok_or(BridgeError::InvalidJson)?;
        let command_type = envelope
            .get("type")
            .and_then(|kind| kind.as_str())
            .ok_or(BridgeError::BadEnvelope)?;
        parse_command_payload(body).ok_or_else(|| BridgeError::InvalidCommand {
            command: command_type.to_owned(),
        })
    })();
    IncomingPost {
        request_id,
        command,
    }
}

/// Serializes an acknowledgement in the HTTP-ack-compatible shape:
/// `{"accepted", "request_id"[, "error"]}`.
pub fn ack_json(request_id: RequestId, accepted: bool, error: Option<&str>) -> String {
    match error {
        Some(error) => serde_json::json!({
            "accepted": accepted,
            "request_id": request_id,
            "error": error,
        })
        .to_string(),
        None => serde_json::json!({
            "accepted": accepted,
            "request_id": request_id,
        })
        .to_string(),
    }
}

/// Wraps an ack for the host → page channel:
/// `{"kind": "ack", "accepted", "request_id"[, "error"]}`.
///
/// Delivered to the page via [`dispatch_script`]; the page correlates it
/// with the post by `request_id`.
pub fn ack_message_json(outcome: &AckOutcome) -> String {
    let mut value =
        serde_json::from_str::<serde_json::Value>(&outcome.to_json()).unwrap_or_default();
    if let Some(object) = value.as_object_mut() {
        object.insert("kind".into(), serde_json::json!("ack"));
    }
    serde_json::to_string(&value).unwrap_or_else(|_| outcome.to_json())
}

/// Converts one engine event to its canonical externally-tagged JSON value
/// with a transport `sequence` sibling.
///
/// Variant shapes are identical to `/api/events` records, so page code can
/// share one event parser between the browser (HTTP) and embedded (webview)
/// modes; only the delivery differs.
pub fn event_json(event: &GameEvent, sequence: EventSeq) -> serde_json::Value {
    let mut value = match event {
        GameEvent::EntityCreated { entity_id } => {
            serde_json::json!({"EntityCreated": {"entity_id": entity_id}})
        }
        GameEvent::EntityDestroyed { entity_id } => {
            serde_json::json!({"EntityDestroyed": {"entity_id": entity_id}})
        }
        GameEvent::ComponentUpdated {
            entity_id,
            type_name,
            json_data,
        } => serde_json::json!({
            "ComponentUpdated": {
                "entity_id": entity_id,
                "type_name": type_name,
                "json_data": embedded_json(json_data),
            }
        }),
        GameEvent::CustomEvent {
            cmd_type,
            json_data,
        } => serde_json::json!({
            "CustomEvent": {
                "cmd_type": cmd_type,
                "json_data": embedded_json(json_data),
            }
        }),
        GameEvent::CommandCompleted {
            request_id,
            command,
            success,
            error,
        } => serde_json::json!({
            "CommandCompleted": {
                "request_id": request_id,
                "command": command,
                "success": success,
                "error": error,
            }
        }),
        GameEvent::EventGap { after, oldest } => serde_json::json!({
            "EventGap": {
                "after": after,
                "oldest": oldest,
            }
        }),
    };
    if let Some(object) = value.as_object_mut() {
        object.insert("sequence".into(), serde_json::json!(sequence));
    }
    value
}

/// Serializes an event batch for the host → page channel:
/// `{"kind": "events", "events": [{…event, "sequence"}, …]}`.
pub fn events_message_json(batch: &[(EventSeq, GameEvent)]) -> String {
    let events: Vec<serde_json::Value> = batch
        .iter()
        .map(|(sequence, event)| event_json(event, *sequence))
        .collect();
    serde_json::json!({"kind": "events", "events": events}).to_string()
}

/// Builds the `evaluate_script` snippet delivering one host → page message.
///
/// The payload (an [`ack_message_json`]/[`events_message_json`] object)
/// arrives as the `detail` of a `CustomEvent("ornis-ipc")` on `window`;
/// JSON is a valid JS expression, so no quoting or escaping layer is needed.
pub fn dispatch_script(message_json: &str) -> String {
    format!("window.dispatchEvent(new CustomEvent(\"ornis-ipc\",{{detail:{message_json}}}));")
}

/// Initialization script installed on every page load (before `onload`).
///
/// Marks embedded mode (`window.ornisShell = {scheme, api}`) and provides
/// a promise-based `window.ornisSend(type, data)` helper over the raw
/// `window.ipc.postMessage` channel: acks and event batches arrive as
/// `ornis-ipc` custom events (see [`dispatch_script`]) and are routed to
/// pending posts by `request_id` or to `window.ornisOnEvents` subscribers.
pub fn init_script() -> &'static str {
    concat!(
        "(function(){",
        "window.ornisShell={scheme:\"ornis\",api:1};",
        "var nextId=1;var pending=new Map();var subs=[];",
        "window.ornisOnEvents=function(fn){subs.push(fn);};",
        "window.ornisSend=function(type,data){",
        "var id=nextId++;",
        "return new Promise(function(resolve,reject){",
        "pending.set(id,{resolve:resolve,reject:reject});",
        "window.ipc.postMessage(JSON.stringify({type:type,request_id:id,data:data||{}}));",
        "});};",
        "window.addEventListener(\"ornis-ipc\",function(ev){",
        "var msg=ev.detail;if(!msg||typeof msg!==\"object\")return;",
        "if(msg.kind===\"ack\"&&typeof msg.request_id===\"number\"){",
        "var p=pending.get(msg.request_id);if(!p)return;pending.delete(msg.request_id);",
        "if(msg.accepted){p.resolve(msg);}else{p.reject(new Error(msg.error||\"rejected\"));}return;}",
        "if(msg.kind===\"events\"&&Array.isArray(msg.events)){",
        "for(var i=0;i<msg.events.length;i++){for(var j=0;j<subs.length;j++){subs[j](msg.events[i]);}}}",
        "});",
        "})();",
    )
}

/// Embeds a `json_data` payload: canonical JSON stays an object, anything
/// else remains a string — so one malformed payload can never corrupt the
/// whole batch (same policy as `/api/events`).
fn embedded_json(json_data: &str) -> serde_json::Value {
    serde_json::from_str(json_data)
        .unwrap_or_else(|_| serde_json::Value::String(json_data.to_owned()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use editor_backend::ipc::EditorCommand;

    #[test]
    fn incoming_ping_queues_with_client_request_id() {
        let mut next = RequestId::new(1);
        let post = parse_incoming(r#"{"type":"ping","request_id":41,"data":{}}"#, &mut next);
        assert_eq!(post.request_id, 41);
        assert_eq!(next, 42, "allocator advances past the client id");
        let (request_id, queued) = post.into_queued();
        assert_eq!(request_id, 41);
        match queued.expect("valid post queues") {
            UiCommand::WithRequestId {
                request_id,
                command,
            } => {
                assert_eq!(request_id, 41);
                assert!(matches!(
                    *command,
                    UiCommand::Custom { cmd_type, .. } if cmd_type == EditorCommand::Ping
                ));
            }
            _ => panic!("expected request-id wrapper"),
        }
    }

    #[test]
    fn incoming_without_request_id_allocates_monotonic_ids() {
        let mut next = RequestId::new(1);
        let first = parse_incoming(r#"{"type":"ping"}"#, &mut next);
        let second = parse_incoming(r#"{"type":"ping"}"#, &mut next);
        assert_eq!(first.request_id, 1);
        assert_eq!(second.request_id, 2);
        // A zero client id is not an id — it allocates like an absent one.
        let third = parse_incoming(r#"{"type":"ping","request_id":0}"#, &mut next);
        assert_eq!(third.request_id, 3);
        assert!(first.command.is_ok());
    }

    #[test]
    fn incoming_set_component_builds_typed_upsert() {
        let mut next = RequestId::new(7);
        let post = parse_incoming(
            r#"{"type":"set_component","data":{"id":5,"component":"Transform","value":{"x":1}}}"#,
            &mut next,
        );
        assert_eq!(post.request_id, 7);
        let (_, queued) = post.into_queued();
        match queued.expect("valid upsert queues") {
            UiCommand::WithRequestId { command, .. } => {
                assert!(matches!(
                    *command,
                    UiCommand::SetComponent { entity_id: 5, .. }
                ));
            }
            _ => panic!("expected request-id wrapper"),
        }
    }

    #[test]
    fn incoming_rejections_are_typed_and_keep_their_id() {
        let mut next = RequestId::new(1);
        let invalid = parse_incoming("not json", &mut next);
        assert_eq!(invalid.request_id, 1);
        assert_eq!(invalid.command.unwrap_err(), BridgeError::InvalidJson);
        let no_type = parse_incoming(r#"{"foo":1}"#, &mut next);
        assert_eq!(no_type.request_id, 2);
        assert_eq!(no_type.command.unwrap_err(), BridgeError::BadEnvelope);
        let bad_type = parse_incoming(r#"{"type":7}"#, &mut next);
        assert_eq!(bad_type.command.unwrap_err(), BridgeError::BadEnvelope);
        // A client id survives even a rejected post, for ack correlation.
        let bad_data = parse_incoming(
            r#"{"type":"set_component","request_id":50,"data":{"id":1}}"#,
            &mut next,
        );
        assert_eq!(bad_data.request_id, 50);
        assert_eq!(
            bad_data.command.unwrap_err(),
            BridgeError::InvalidCommand {
                command: "set_component".into()
            }
        );
        assert_eq!(next, RequestId::new(51));
    }

    #[test]
    fn ack_outcomes_serialize_like_the_http_ack() {
        let accepted = AckOutcome::Accepted {
            request_id: RequestId::new(7),
        };
        assert!(accepted.accepted());
        assert_eq!(accepted.request_id(), RequestId::new(7));
        let value = serde_json::from_str::<serde_json::Value>(&accepted.to_json())
            .expect("accepted ack is valid JSON");
        assert_eq!(
            value,
            serde_json::json!({"accepted": true, "request_id": 7})
        );

        let rejected = AckOutcome::Rejected {
            request_id: RequestId::new(8),
            error: "bad request".into(),
        };
        assert!(!rejected.accepted());
        let value = serde_json::from_str::<serde_json::Value>(&rejected.to_json())
            .expect("rejected ack is valid JSON");
        assert_eq!(
            value,
            serde_json::json!({"accepted": false, "request_id": 8, "error": "bad request"})
        );
    }

    #[test]
    fn ack_message_marks_kind_for_page_routing() {
        let message =
            serde_json::from_str::<serde_json::Value>(&ack_message_json(&AckOutcome::Accepted {
                request_id: RequestId::new(3),
            }))
            .expect("ack message is valid JSON");
        assert_eq!(message["kind"], "ack");
        assert_eq!(message["accepted"], true);
        assert_eq!(message["request_id"], 3);
    }

    #[test]
    fn events_keep_api_shapes_and_add_sequence() {
        let batch = vec![
            (EventSeq::new(9), GameEvent::EntityCreated { entity_id: 4 }),
            (
                EventSeq::new(10),
                GameEvent::CommandCompleted {
                    request_id: RequestId::new(4),
                    command: "set_component".into(),
                    success: true,
                    error: None,
                },
            ),
            (
                EventSeq::new(11),
                GameEvent::CustomEvent {
                    cmd_type: "status".into(),
                    json_data: r#"{"v":7}"#.into(),
                },
            ),
        ];
        let message = serde_json::from_str::<serde_json::Value>(&events_message_json(&batch))
            .expect("events message is valid JSON");
        assert_eq!(message["kind"], "events");
        assert_eq!(message["events"][0]["sequence"], 9);
        assert_eq!(message["events"][0]["EntityCreated"]["entity_id"], 4);
        assert_eq!(message["events"][1]["sequence"], 10);
        assert_eq!(message["events"][1]["CommandCompleted"]["request_id"], 4);
        assert_eq!(
            message["events"][2]["CustomEvent"]["json_data"],
            serde_json::json!({"v": 7})
        );
    }

    #[test]
    fn events_escape_text_and_keep_malformed_payloads_valid_json() {
        let batch = vec![(
            EventSeq::new(1),
            GameEvent::ComponentUpdated {
                entity_id: 3,
                type_name: "Transform\"\n".into(),
                json_data: "not-json".into(),
            },
        )];
        let message = serde_json::from_str::<serde_json::Value>(&events_message_json(&batch))
            .expect("escaped batch stays valid JSON");
        assert_eq!(
            message["events"][0]["ComponentUpdated"]["type_name"],
            "Transform\"\n"
        );
        assert_eq!(
            message["events"][0]["ComponentUpdated"]["json_data"],
            "not-json"
        );
    }

    #[test]
    fn dispatch_script_delivers_payload_as_ornis_ipc_detail() {
        let script = dispatch_script(r#"{"kind":"ack","request_id":1}"#);
        assert!(script.contains("ornis-ipc"), "page listens on ornis-ipc");
        assert!(script.contains(r#"{"kind":"ack","request_id":1}"#));
        // The payload is spliced as a JS expression, not a quoted string.
        assert!(!script.contains("\\\"kind\\\""));
    }

    #[test]
    fn init_script_exposes_shell_marker_and_send_helper() {
        let script = init_script();
        assert!(script.contains("window.ornisShell"));
        assert!(script.contains("window.ornisSend"));
        assert!(script.contains("window.ipc.postMessage"));
        assert!(script.contains("ornis-ipc"));
    }
}
