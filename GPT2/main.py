from transformers import (
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    TextDataset,
    GPT2LMHeadModel,
    TrainingArguments,
    Trainer,
    pipeline)

from decoding_methods import set_generator, greedy_search, standard_search, beam_search, random_sampling, top_k_search, top_p_search

import os
from enum import Enum
from tqdm import tqdm

class DecodingMethods(Enum):
    GREEDY_SEARCH = 0
    BEAM_SEARCH = 1
    TOP_K_SAMPLING = 2
    TOP_P_SAMPLING = 3
    STANDARD_SEARCH = 4
    RANDOM_SAMPLING = 5



def main(train_path, test_path, val_path, num_of_samples=5, decoding_method = DecodingMethods.TOP_K_SAMPLING):
    # Get tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding=True, truncation=True, return_tensors="pt")

    # Check if it worked
    print('vocabulary size: %d, max squence length: %d' % (tokenizer.vocab_size, tokenizer.model_max_length))
    print('tokenize sequence:', tokenizer('Republicans and Democrats have both created our economic problems'))

    # Data collator for separating into batches
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create all datasets
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=128)
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=128)
    val_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=val_path,
        block_size=128)

    # Download the headless GPT-2 model for transfer learning
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Expand the tokenizer and model with special tokens
    # Add beginning of sentence, end of sentence and padding tokens for a better dataset
    special_tokens_dict = {'bos_token': '<BOS>', 'eos_token': '<EOS>', 'pad_token': '<PAD>'}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    # Create an output folder
    output_folder = "../output/gpt2/"
    if os.path.isdir(output_folder):
        os.makedirs(output_folder)
        os.mkdir(output_folder + "model")
        os.mkdir(output_folder + "text")

    # Training the transformer model
    training_args = TrainingArguments(
        output_dir = output_folder + "model", # the output directory for the model predictions and checkpoints
        overwrite_output_dir = True, # overwrite the content of the output directory
        per_device_train_batch_size = 4, # the batch size for training
        per_device_eval_batch_size = 4, # the batch size for evaluation
        learning_rate = 5e-5, # defaults to 5e-5
        num_train_epochs = 3, # total number of training epochs to perform
    )

    trainer = Trainer(
        model = model,
        args = training_args,
        data_collator=data_collator,
        train_dataset = train_dataset,
        eval_dataset = test_dataset
    )

    trainer.train()
    trainer.save_model()

    # Predicting with the model

    set_generator(output_folder)

    generator_methods = {
        DecodingMethods.GREEDY_SEARCH: greedy_search,
        DecodingMethods.STANDARD_SEARCH: standard_search,
        DecodingMethods.BEAM_SEARCH: beam_search,
        DecodingMethods.TOP_K_SAMPLING: top_k_search,
        DecodingMethods.TOP_P_SAMPLING: top_p_search,
        DecodingMethods.RANDOM_SAMPLING: random_sampling
    }

    with open(output_folder + "text/output.txt", "w") as f:
        print("Using the decoding method: " + decoding_method)
        for _ in tqdm(range(num_of_samples)):
            f.write(generator_methods[decoding_method]())

if __name__=="__main__":
    train_path = "../train.txt"
    test_path = "../test.txt"
    val_path = "../validation.txt"
    main(train_path, test_path, val_path, num_of_samples=10, decoding_method=DecodingMethods.BEAM_SEARCH)
