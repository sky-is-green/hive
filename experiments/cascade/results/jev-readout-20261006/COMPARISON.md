# J-Bench comparison (frozen test split)
| readout | kind | n | AUC | best BA | acc@0.5 | FA/FR@0.5 | no-leak thr | no-leak acc | ECE |
|---|---|---|---|---|---|---|---|---|---|
| qwen35-4b | prompted | 116 | 0.494 | 0.546 | 0.353 | 70/5 | 0.91 | 0.750 | 0.429 |
| scion-own-prompted | prompted | 30 | 0.591 | 0.750 | 0.767 | 7/0 | 0.71 | 0.333 | 0.226 |
| scion35b-prompted | prompted | 116 | 0.903 | 0.925 | 0.862 | 15/1 | 0.70 | 0.776 | 0.320 |
| strata-other-prompted | prompted | 30 | 0.103 | 0.552 | 0.933 | 1/1 | 0.70 | 0.133 | 0.364 |
| intern-0.8b | trained-judge | 116 | 0.774 | 0.759 | 0.629 | 37/6 | 0.88 | 0.793 | 0.302 |
| intern-2b | trained-judge | 116 | 0.932 | 0.856 | 0.810 | 20/2 | 0.91 | 0.802 | 0.175 |
| intern-4b | trained-judge | 116 | 0.979 | 0.937 | 0.897 | 11/1 | 0.97 | 0.871 | 0.094 |
| tinyjev-1.7b | trained-judge | 116 | 0.890 | 0.862 | 0.741 | 26/4 | nan | 0.000 | 0.256 |
| qwen3-1.7b depth 0.5 | probe | 116 | 0.915 | 0.862 | 0.672 | 37/1 | nan | 0.000 | 0.311 |
| qwen3-1.7b depth 0.75 | probe | 116 | 0.988 | 0.954 | 0.888 | 12/1 | 0.94 | 0.948 | 0.133 |
| qwen3-1.7b depth 1 | probe | 116 | 0.950 | 0.908 | 0.845 | 18/0 | nan | 0.000 | 0.183 |
| qwen35-0.8b depth 0.5 | probe | 116 | 0.792 | 0.759 | 0.716 | 24/9 | nan | 0.000 | 0.235 |
| qwen35-0.8b depth 0.75 | probe | 116 | 0.924 | 0.856 | 0.655 | 40/0 | nan | 0.000 | 0.356 |
| qwen35-0.8b depth 1 | probe | 116 | 0.795 | 0.713 | 0.578 | 45/4 | nan | 0.000 | 0.375 |
| qwen35-4b depth 0.5 | probe | 116 | 0.899 | 0.833 | 0.767 | 24/3 | nan | 0.000 | 0.218 |
| qwen35-4b depth 0.75 | probe | 116 | 0.946 | 0.914 | 0.845 | 17/1 | nan | 0.000 | 0.172 |
| qwen35-4b depth 1 | probe | 116 | 0.960 | 0.937 | 0.871 | 15/0 | nan | 0.000 | 0.157 |
| scion35b depth 1 | probe | 116 | 0.999 | 0.994 | 0.931 | 8/0 | 0.95 | 0.974 | 0.151 |
