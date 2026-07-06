# bayesite-idata

Bayescycle run-directory to ArviZ DataTree/NetCDF exporter.

This distribution exposes the `bayesite-idata` CLI and the `bayesite_idata`
import package. It intentionally excludes the plotting stack (`arviz-plots`,
`arviz-stats`, `matplotlib`) and writes `.nc` / `.idata` files for downstream
consumers such as `bayesite-viz`.
