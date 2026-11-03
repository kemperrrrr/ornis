atomicAdd(&atomic_storage_0_2[1u], 4u)
workgroup_scratch[3u] = 9u
atomicAdd(&workgroup_atomics[3u], workgroup_scratch[3u])
workgroupBarrier()
storageBarrier()
