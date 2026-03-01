import sglang as sgl

# 1. Connect to the local backend we just started
sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))

# 2. Define the workflow using the @sgl.function decorator
@sgl.function
def article_evaluator(s, article_text):
    # Setup the shared system and user prompt
    s += sgl.system("You are an expert editorial assistant.")
    s += sgl.user(f"Read this article:\n{article_text}\n\nIs this article related to Technology?")
    
    # SGLang will force the model to pick ONLY 'yes' or 'no'
    s += sgl.assistant(sgl.select("is_tech", choices=["yes", "no"]))
    
    # Native Python control flow based on LLM output
    if s["is_tech"] == "no":
        s += sgl.user("Provide a 1-sentence summary of what it is actually about.")
        s += sgl.assistant(sgl.gen("summary", max_tokens=50))
        return
        
    # --- THE MAGIC OF PARALLEL BRANCHING ---
    # If it is about tech, we fork the execution into 3 parallel paths.
    aspects = ["grammar", "technical accuracy", "tone"]
    forks = s.fork(len(aspects))
    
    for f, aspect in zip(forks, aspects):
        f += sgl.user(f"Evaluate the {aspect} of the article. Provide a score out of 10.")
        
        # We can force the output to match a specific Regex pattern (JSON)
        # SGLang's engine builds a finite state machine to guarantee this format instantly.
        json_schema = r'\{"score": \d{1,2}, "explanation": "[^"]+"\}'
        f += sgl.assistant(sgl.gen(f"{aspect}_evaluation", regex=json_schema))

# 3. Execute the workflow
if __name__ == "__main__":
    sample_article = "SGLang introduces RadixAttention to drastically improve KV cache reuse across multi-turn prompts."
    
    # Run the compiled SGLang function
    state = article_evaluator.run(article_text=sample_article)
    
    print(f"Is it about Tech? {state['is_tech']}")
    
    # If the forks ran, we can extract their specific generated variables
    if state["is_tech"] == "yes":
        print("\n--- Parallel Evaluations ---")
        # state.children contains the data from the parallel forks
        for child, aspect in zip(state.children, ["grammar", "technical accuracy", "tone"]):
            print(f"{aspect.upper()}: {child[f'{aspect}_evaluation']}")