CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_org.py --data_path './code_description_gpt_as_refer_reordering_chunk/50' \
												--model_dir 't5base_exp3/checkpoint-13000' \
												--output_dir 't5base_exp3' \
												--train_mode 'org_att_parallel_vocab' \
												--loss_mode 'label' \
												--model_name 't5-base'
