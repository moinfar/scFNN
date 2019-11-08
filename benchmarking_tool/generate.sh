python run.py cell_cycle_hq -D -S 123 generate -o files/test\ benchs/cell_cycle_hq.csv cell-cycle --rm-ercc --rm-mt --rm-lq

python run.py clustering_cortex -D -S 132 generate -o files/test\ benchs/clustering_cortex.csv clustering -d CORTEX_3005
python run.py clustering_pollen_lq -D -S 123 generate -o files/test\ benchs/clustering_pollen_lq.csv clustering -d POLLEN-LQ
python run.py clustering_pollen_hq -D -S 123 generate -o files/test\ benchs/clustering_pollen_hq.csv clustering -d POLLEN-HQ
python run.py clustering_baron_human -D -S 123 generate -o files/test\ benchs/clustering_baron_human.csv clustering -d BARON-HUMAN
python run.py clustering_baron_mouse -D -S 123 generate -o files/test\ benchs/clustering_baron_mouse.csv clustering -d BARON-MOUSE

python run.py random_mask_cortex_1 -D -S 123 generate -o files/test\ benchs/random_mask_cortex_1.csv random-mask -d CORTEX_3005 --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_cortex_2 -D -S 123 generate -o files/test\ benchs/random_mask_cortex_2.csv random-mask -d CORTEX_3005 --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_10xPBMC4k_1 -D -S 123 generate -o files/test\ benchs/random_mask_10xPBMC4k_1.csv random-mask -d 10xPBMC4k --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_10xPBMC4k_2 -D -S 123 generate -o files/test\ benchs/random_mask_10xPBMC4k_2.csv random-mask -d 10xPBMC4k --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_cell_cycle_1 -D -S 123 generate -o files/test\ benchs/random_mask_cell_cycle_1.csv random-mask -d CELL_CYCLE --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_cell_cycle_2 -D -S 123 generate -o files/test\ benchs/random_mask_cell_cycle_2.csv random-mask -d CELL_CYCLE --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_baron_mouse_1 -D -S 123 generate -o files/test\ benchs/random_mask_baron_mouse_1.csv random-mask -d BARON-MOUSE --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_baron_mouse_2 -D -S 123 generate -o files/test\ benchs/random_mask_baron_mouse_2.csv random-mask -d BARON-MOUSE --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_baron_human_1 -D -S 123 generate -o files/test\ benchs/random_mask_baron_human_1.csv random-mask -d BARON-HUMAN --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_baron_human_2 -D -S 123 generate -o files/test\ benchs/random_mask_baron_human_2.csv random-mask -d BARON-HUMAN --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_cite_cd8_1 -D -S 123 generate -o files/test\ benchs/random_mask_cite_cd8_1.csv random-mask -d CITE-CD8 --dropout-count 1000 --hvg-frac 1 --min-expression 10
python run.py random_mask_cite_cd8_2 -D -S 123 generate -o files/test\ benchs/random_mask_cite_cd8_2.csv random-mask -d CITE-CD8 --dropout-count 1000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_pollen_hq_1 -D -S 123 generate -o files/test\ benchs/random_mask_pollen_hq_1.csv random-mask -d POLLEN-HQ --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_pollen_hq_2 -D -S 123 generate -o files/test\ benchs/random_mask_pollen_hq_2.csv random-mask -d POLLEN-HQ --dropout-count 10000 --hvg-frac 0.1 --min-expression 10
python run.py random_mask_pollen_lq_1 -D -S 123 generate -o files/test\ benchs/random_mask_pollen_lq_1.csv random-mask -d POLLEN-LQ --dropout-count 10000 --hvg-frac 1 --min-expression 10
python run.py random_mask_pollen_lq_2 -D -S 123 generate -o files/test\ benchs/random_mask_pollen_lq_2.csv random-mask -d POLLEN-LQ --dropout-count 10000 --hvg-frac 0.1 --min-expression 10

python run.py down_sample_cortex_1 -D -S 123 generate -o files/test\ benchs/down_sample_cortex_1.csv down-sample -d CORTEX_3005 --read-ratio 0.1
python run.py down_sample_cortex_2 -D -S 123 generate -o files/test\ benchs/down_sample_cortex_2.csv down-sample -d CORTEX_3005 --read-ratio 0.5
python run.py down_sample_10xPBMC4k_1 -D -S 123 generate -o files/test\ benchs/down_sample_10xPBMC4k_1.csv down-sample -d 10xPBMC4k --read-ratio 0.1
python run.py down_sample_10xPBMC4k_2 -D -S 123 generate -o files/test\ benchs/down_sample_10xPBMC4k_2.csv down-sample -d 10xPBMC4k --read-ratio 0.5
python run.py down_sample_cell_cycle_1 -D -S 123 generate -o files/test\ benchs/down_sample_cell_cycle_1.csv down-sample -d CELL_CYCLE --read-ratio 0.1
python run.py down_sample_cell_cycle_2 -D -S 123 generate -o files/test\ benchs/down_sample_cell_cycle_2.csv down-sample -d CELL_CYCLE --read-ratio 0.5
python run.py down_sample_baron_human_1 -D -S 123 generate -o files/test\ benchs/down_sample_baron_human_1.csv down-sample -d BARON-HUMAN --read-ratio 0.1
python run.py down_sample_baron_human_2 -D -S 123 generate -o files/test\ benchs/down_sample_baron_human_2.csv down-sample -d BARON-HUMAN --read-ratio 0.5
python run.py down_sample_baron_mouse_1 -D -S 123 generate -o files/test\ benchs/down_sample_baron_mouse_1.csv down-sample -d BARON-MOUSE --read-ratio 0.1
python run.py down_sample_baron_mouse_2 -D -S 123 generate -o files/test\ benchs/down_sample_baron_mouse_2.csv down-sample -d BARON-MOUSE --read-ratio 0.5
python run.py down_sample_cite_cd8_1 -D -S 123 generate -o files/test\ benchs/down_sample_cite_cd8_1.csv down-sample -d CITE-CD8 --read-ratio 0.1
python run.py down_sample_cite_cd8_2 -D -S 123 generate -o files/test\ benchs/down_sample_cite_cd8_2.csv down-sample -d CITE-CD8 --read-ratio 0.5
python run.py down_sample_pollen_hq_1 -D -S 123 generate -o files/test\ benchs/down_sample_pollen_hq_1.csv down-sample -d POLLEN-HQ --read-ratio 0.1
python run.py down_sample_pollen_hq_2 -D -S 123 generate -o files/test\ benchs/down_sample_pollen_hq_2.csv down-sample -d POLLEN-HQ --read-ratio 0.5
python run.py down_sample_pollen_lq_1 -D -S 123 generate -o files/test\ benchs/down_sample_pollen_lq_1.csv down-sample -d POLLEN-LQ --read-ratio 0.1
python run.py down_sample_pollen_lq_2 -D -S 123 generate -o files/test\ benchs/down_sample_pollen_lq_2.csv down-sample -d POLLEN-LQ --read-ratio 0.5

python run.py paired_data_pollen -D -S 123 generate -o files/test\ benchs/paired_data_pollen.csv paired-data -d POLLEN

python run.py cite_seq_pbmc -D -S 123 generate -o files/test\ benchs/cite_seq_pbmc.csv cite-seq -d CITE-PBMC
python run.py cite_seq_cbmc -D -S 123 generate -o files/test\ benchs/cite_seq_cbmc.csv cite-seq -d CITE-CBMC
python run.py cite_seq_cd8 -D -S 123 generate -o files/test\ benchs/cite_seq_cd8.csv cite-seq -d CITE-CD8
