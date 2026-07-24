const { chromium } = require('playwright');
const fs = require('fs');
const path = require('path');
const http = require('http');

(async () => {
  const viewportW = parseInt(process.argv[2] || '1920');
  const viewportH = parseInt(process.argv[3] || '1080');
  const htmlPath = path.resolve(process.argv[4] || 'editor_test.html');
  const outputPath = process.argv[5] || 'chromium_layout.json';

  // Start local HTTP server
  const server = http.createServer((req, res) => {
    const filePath = path.join(path.dirname(htmlPath), req.url === '/' ? path.basename(htmlPath) : req.url);
    try {
      const content = fs.readFileSync(filePath);
      const ext = path.extname(filePath);
      const mime = ext === '.css' ? 'text/css' :
                   ext === '.js' ? 'application/javascript' :
                   ext === '.html' ? 'text/html' : 'application/octet-stream';
      res.writeHead(200, { 'Content-Type': mime });
      res.end(content);
    } catch (e) {
      res.writeHead(404);
      res.end('Not found');
    }
  });

  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const port = server.address().port;
  const url = `http://127.0.0.1:${port}/`;

  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: viewportW, height: viewportH } });

  await page.goto(url, { waitUntil: 'load', timeout: 15000 });

  // Inject :root font-size to match our engine's rem_base
  await page.evaluate(() => {
    const s = document.createElement('style');
    s.textContent = ':root { font-size: 12px !important; }';
    document.head.appendChild(s);
  });
  await page.waitForTimeout(200);

  const data = await page.evaluate((viewport) => {
    const skipTags = new Set(['html','head','script','style','meta','link','title']);
    const all = document.querySelectorAll('*');
    const nodes = [];

    function parentChain(el) {
      const c = [];
      let cur = el.parentElement;
      while (cur) {
        const t = cur.tagName.toLowerCase();
        const cls = cur.className && cur.className.trim()
          ? '.' + cur.className.trim().split(/\s+/).join('.')
          : '';
        c.push(t + cls);
        cur = cur.parentElement;
      }
      return c;
    }

    const keepProps = [
      'display','position','width','height','min-width','min-height',
      'max-width','max-height',
      'margin','margin-left','margin-right','margin-top','margin-bottom',
      'padding','padding-left','padding-right','padding-top','padding-bottom',
      'font-size','font-family','font-weight','line-height',
      'text-align','color','background-color','background-image',
      'border','border-radius','box-sizing',
      'overflow','overflow-x','overflow-y',
      'flex-direction','flex-wrap','flex-grow','flex-shrink','flex-basis',
      'align-items','align-content','justify-content','gap',
      'left','top','right','bottom',
      'opacity','visibility','pointer-events','cursor',
      'white-space','text-overflow','letter-spacing','word-spacing',
      'user-select','object-fit','transform','transform-origin',
      'fill','stroke','stroke-width',
    ];

    for (const el of all) {
      const tag = el.tagName.toLowerCase();
      if (skipTags.has(tag)) continue;

      const r = el.getBoundingClientRect();
      const s = window.getComputedStyle(el);
      const styles = {};
      for (const p of keepProps) {
        const v = s.getPropertyValue(p);
        if (v && v !== 'none' && v !== 'normal') {
          styles[p] = v;
        }
      }

      nodes.push({
        tag,
        dom_class: el.className?.trim() || null,
        dom_id: el.id || null,
        label: tag + (el.className?.trim() ? '.' + el.className.trim().split(/\s+/).join('.') : ''),
        rect: {
          x: Math.round(r.x * 100) / 100,
          y: Math.round(r.y * 100) / 100,
          width: Math.round(r.width * 100) / 100,
          height: Math.round(r.height * 100) / 100,
        },
        styles,
        parent_chain: parentChain(el),
      });
    }
    return {
      viewport: { width: viewport.w, height: viewport.h },
      node_count: nodes.length,
      nodes,
    };
  }, { w: viewportW, h: viewportH });

  fs.writeFileSync(outputPath, JSON.stringify(data, null, 2));
  console.log(`wrote ${outputPath}  (viewport ${viewportW}x${viewportH}, ${data.node_count} nodes)`);

  await browser.close();
  server.close();
})();
