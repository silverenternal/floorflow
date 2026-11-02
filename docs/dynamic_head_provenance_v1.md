# Dynamic head provenance

The local adapters are pinned to the following source revisions. Only model
concepts and operator structure are adapted; training code and datasets remain
inside their upstream repositories.

| Local source | Revision | License | Adapted component |
| --- | --- | --- | --- |
| `third_party/EoMT` | `aa017c34afbe023de9462b17ae6067b52fef9867` | MIT | Learned query/mask head and query-to-feature mask projection |
| `third_party/MiTA` | `95460f703ef5bd5d814c45a6ff52d9582207d01c` | MIT | Routed cross-attention and dynamic query budget |
| `third_party/dynamic-sparse-flash-attention` | `266f36a914216c439f5aa63e12b00fa8489fde95` | MIT | Sparse routing diagnostics and budgeted candidate selection |

The Microsoft DynamicHead repository was not available during this revision
because the remote TLS handshake failed. The local implementation therefore
does not claim code from that repository. It implements the same scale,
spatial and task routing idea using the pinned MIT sources above and generated
depthwise kernels from pooled sample context.

Integration rules:

1. Third-party code never owns CubiCasa preprocessing or evaluation.
2. Every dynamic head starts as a fixed-head-compatible zero residual.
3. Routing tests run on CPU before GPU training.
4. Checkpoints record the head variant and candidate kernel configurations.
5. Test labels never drive routing or checkpoint selection.
