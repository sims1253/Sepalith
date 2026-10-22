# Whitespace near-misses and decode-time canonicalisation (arm A 536)

Measurement only. The official scorer is unchanged. Variants are fixed in code, run on TRAIN first, then on DEV.
L0 to L3a change only the prediction and use only prompt-side information. G1 to G3 game the metric.

| Variant | TRAIN exact rate (census, T=1.0) | TRAIN nofp | DEV250 edit /200 | DEV250 no-op /50 | score250 | DEV250 nofp | DEV75 edit /43 | DEV75 nofp /32 | DEV75 valid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw | 0.4323 (51898/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| L0 | 0.4316 (51814/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| L1 | 0.4323 (51899/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| L2 | 0.4323 (51901/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| L2s | 0.4222 (50685/120048) | 12 | 78 | 40 | 118 | 10 | 26 | 8 | 74 |
| L3 | 0.4255 (51082/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| L3a | 0.4181 (50188/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |
| G1 | 0.4295 (51559/120048) | 12 | 79 | 40 | 119 | 10 | 26 | 8 | 74 |
| G2 | 0.4349 (52212/120048) | 12 | 80 | 38 | 118 | 12 | 27 | 8 | 74 |
| G3 | 0.4344 (52145/120048) | 12 | 79 | 38 | 117 | 12 | 27 | 8 | 74 |

Whitespace-only misses (valid wrong outputs whose text matches after removing whitespace):

- train: {'edit_case_ws_only_vs_gold': 2464, 'edit_case_pred_equals_old_mod_ws': 59}; types {'line_wrapping': 1401, 'indentation': 401, 'leading_blank_lines': 370, 'indent:uniform_offset': 310, 'interior_blank_lines': 208, 'intra_line_whitespace': 126, 'indent:partial': 84, 'trailing_whitespace': 27, 'indent:tabs_vs_spaces': 17, 'indent:scale': 7, 'trailing_blank_lines': 1}
- dev250: {'edit_case_ws_only_vs_gold': 1, 'noop_case_ws_only_suggestion': 6}; types {'line_wrapping': 3, 'intra_line_whitespace': 3, 'indent:uniform_offset': 1, 'indentation': 1}
- dev75: {'edit_case_ws_only_vs_gold': 1}; types {'line_wrapping': 1}
