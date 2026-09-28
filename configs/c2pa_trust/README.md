# Pinned C2PA trust lists

These public certificate lists let the provenance checker evaluate signer and
timestamp trust without downloading anything while it inspects an image.

- Source repository: `c2pa-org/conformance-public`
- Pinned commit: `5a94626972e693dcfef53d59da56183ea8d3f8e5`
- Commit date: `2026-09-22T19:17:45Z`
- Snapshot date: `2026-09-28`
- Licence: CC BY 4.0
- Attribution: Coalition for Content Provenance and Authenticity (C2PA)

Files:

- `C2PA-TRUST-LIST.pem`
  - SHA-256: `75cacc98b79ecac33713c7ecfb58d4a0ef383f3c1f886e7409f9e37e8664aea5`
  - Source: <https://github.com/c2pa-org/conformance-public/blob/5a94626972e693dcfef53d59da56183ea8d3f8e5/trust-list/C2PA-TRUST-LIST.pem>
- `C2PA-TSA-TRUST-LIST.pem`
  - SHA-256: `c688d3555f4a2f1f8d663472bbd37888ff234abdd234c25934c0f9292e4eb5c9`
  - Source: <https://github.com/c2pa-org/conformance-public/blob/5a94626972e693dcfef53d59da56183ea8d3f8e5/trust-list/C2PA-TSA-TRUST-LIST.pem>

Do not update these files automatically during inference. A future update must
pin a new commit, record new hashes, run the C2PA interoperability tests, and be
reviewed as a deliberate dependency change.
