
generator = None

def set_generator(output_folder):
    generator = pipeline("text-generation", tokenizer="gpt2", model=output_folder + "model")

def greedy_search():
    if generator is None:
        print("Error: generator not set")
        return
    print("Error: greedy search is not implemented yet")

def standard_search():
    if generator is None:
        print("Error: generator not set")
        return

def beam_search():
    if generator is None:
        print("Error: generator not set")
        return
    return generator('',
                     max_length=40,
                     num_beams=5)

def random_sampling():
    if generator is None:
        print("Error: generator not set")
        return
    return generator('',
                     max_length=40,
                     top_k=0,
                     do_sample=True,
                     temperature=0.7)

def top_k_search():
    if generator is None:
        print("Error: generator not set")
        return
    return generator('',
                     max_length=40,
                     top_k=40,
                     do_sample=True)

def top_p_search():
    if generator is None:
        print("Error: generator not set")
        return
    return generator('',
                     max_length=40,
                     top_k=0,
                     top_p=0.92,
                     do_sample=True)
