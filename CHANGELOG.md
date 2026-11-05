# Changelog

## Unreleased

- Fix NB101 decoding of cells with fewer than 7 nodes. Such cells are stored padded with edgeless `output` nodes; the decoder made the last (padding) slot the output, so the cell was rejected as disconnected. Trailing padding is now stripped before `ModelSpec`, so these cells decode to their original valid architecture. Sampled NB101 valid rates rise accordingly; 7-node cells are unaffected.
