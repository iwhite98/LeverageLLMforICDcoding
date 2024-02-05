CUDA_VISIBLE_DEVICES=0,1,2,3 python finetune_org_ref.py --data_path './code_description_gpt_as_refer_reordering_chunk/50' \
												--output_dir 't5base_exp3' \
												--batch_size 4 \
												--train_mode 'org_att_parallel_vocab' \
												--loss_mode 'label' \
												--model_name 't5-base' \
												--num_epoch 30
