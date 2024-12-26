# Deep Speech-based-Automatic-Speech-Recognition
In this repo we will use DeepSpeech model to generate Automatic Speech Recognition (ASR) / transcription from Noisy Speech data  which is less in quantity (02 challenges)


![LSTM3-chain](https://github.com/user-attachments/assets/4d00d2f1-3884-4366-b4d3-f37ee968d931)



![Parallelism](https://github.com/user-attachments/assets/3e8303fe-3d33-465e-8e2e-126450d1b197)


![rnn_fig-624x598](https://github.com/user-attachments/assets/9bc3c578-e530-4bbf-b854-78b8ce3e620e)



# Start Setup
`
mkdir Deepspeech
`

`
cd Deepspeech
`

`
sudo apt install git
`
Now we will clone the mozilla deepspeech folder

`
git clone https://github.com/mozilla/DeepSpeech.git
`

Now we will download the model , deepspeech document

`
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm     
`
Download original scorer file original 


`
curl -LO https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer  
`

Download checkpoint directory

`
wget https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-checkpoint.tar.gz   
`

`
tar -xzvf deepspeech-0.9.3-checkpoint.tar.gz 
`

Unzip checkpoint directory

`
cp -r ./DeepSpeech/training/deepspeech_training ./DeepSpeech/ 
`

Now make a directory , now we will make our own scorer file,native client is used to download models


`mkdir native_client
`

`cd native_client
`

`
curl -L https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/native_client.amd64.cuda.linux.tar.xz -o
`

`
native_client.amd64.cuda.linux.tar.xz && tar -Jxvf 
`

`native_client.amd64.cuda.linux.tar.xz
`
`
cd ../
`
# Language model download

`git clone https://github.com/kpu/kenlm.git   
`
dependies of kenlm for ubuntu

`
sudo apt-get install libboost-program-options-dev  
`
`
sudo apt-get install build-essential libboost-all-dev cmake zlib1g-dev libbz2-dev liblzma-dev
`

`
mkdir -p kenlm/build
`
`
cd kenlm/build
`
`
cmake ..
`
we are building kenlm model , ngram approach , cmake checks threads , if threads exist then we can perfrom parallel processing

`
make -j 2  
`

`
cd ../../
`

`
mkdir finetuned_checkpoints
`

# Create environment

`
conda create -n deepspeech-env python=3.7
`

Activate environment

`
conda activate deepspeech-env
`

# Install all dependencies 

`pip install tensorflow-gpu==1.15.4`
`conda install pandas`
`pip install deepspeech==0.9.3`
`pip install deepspeech-gpu`
`pip install protobuf==3.20`
`pip install optuna progressbar2 ds_ctcdecoder attrdict pyxdg semver resampy pyogg pandas`


for full gpu allocation , model takes the gpu fully


`
export TF_FORCE_GPU_ALLOW_GROWTH=true  
`

VERSION   : we need version 0.9 because we have checkpoints of 0.9.3 but it downloads 10 so we need to change it manually  inside version.txt files inside ./DeepSpeech/deepspeech_training/VERSION 

# Custom Scorer File

Now create custom scorer file , it will see what values of alpha and beta are good. It automatically adjusts the values. alpha (acoustic model=preference to asr) and beta (language model=preference to language model). In noisy  and less data (command based instructions) , we prefer whatever model has heard it belives, so we give higher preference to alpha, , so we keep alpha higher

`
python3 ./DeepSpeech/lm_optimizer.py --test_files ./fulltext.csv --checkpoint_dir ./deepspeech-0.9.3-checkpoint --alphabet_config_path ./DeepSpeech/data/alphabet.txt --n_trials 1 --lm_alpha_max 0.5 --lm_beta_max 1 --use_allow_growth true 
`


`mkdir scorer
`


Now create a binary file that contains 5000 sequences of 5 words together. We kept it five words because we have call signs containing 5 words. As a result it  perfroms quantizing 


`python3 DeepSpeech/data/lm/generate_lm.py --input_txt transcriptions_text.txt --output_dir scorer/ --top_k 5000 --kenlm_bins kenlm/build/bin --arpa_order 5 --max_arpa_memory "85%" --arpa_prune "0|0|1" --binary_a_bits 255 --binary_q_bits 8 --binary_type trie --discount_fallback  
`

To activate locked packages  files in ubuntu

`
chmod +x ./native_client/generate_scorer_package `


Now we will get our scorer file for our own dataset with the name of original_scorer with adjusted alpha beta parameters
./native_client/generate_scorer_package  --alphabet /DeepSpeech/data/alphabet.txt   --lm ./scorer/lm.binary   --vocab ./scorer/vocab-5000.txt   --package ./scorer/original_scorer  --default_alpha 1.04   --default_beta 0.93 --force_bytes_output_mode True   

# Training
Now we will starts training the model on our own scorer file and own data

`python3 DeepSpeech/DeepSpeech.py  --train_files train.csv  --dev_files val.csv  --test_files test.csv  --save_checkpoint_dir finetuned_checkpoints --alphabet_config_path DeepSpeech/data/alphabet.txt  --load_checkpoint_dir deepspeech-0.9.3-checkpoint  --reduce_lr_on_plateau true  --plateau_epochs 2 --plateau_reduction 0.05  --early_stop true  --es_epochs 10 --use_allow_growth true --load_cudnn True 
`










