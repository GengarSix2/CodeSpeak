# nohup bash codebert_prompt_batch_detection.sh > codebert_prompt_batch_detection.log

for i in $( seq 1 3 )
do
  # split dataset
  python generate_dataset.py

  # run model
  python codebert_prompt.py --label_category 2 \
  --num_train_epochs 20 \
  --train_batch_size 16 \
  --valid_batch_size 16 \
  --test_batch_size 16 \
  --patience 10 \
  --data_dir ./datasets \
  --model_name_or_path ./models/codebert-base \
  --config_name ./models/codebert-base \
  --tokenizer_name ./models/codebert-base \
  --output_dir ./saved-data/codebert-prompt \
  --output_model_dir ./saved-data/codebert-prompt \
  --train_filename ./datasets \
  --valid_filename ./datasets \
  --test_filename ./datasets \
  --do_train \
  --do_test \
  --soft_prompt
done


  
