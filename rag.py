from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import torch
import re

# ----- RAG Setup -----
# Retrieval Augmented Generation (RAG) architecture.
# Dataset for queries: msmacro v2 queries)
# Retriever: Pyserini with pre-built MSMACRO index
# LLM: Cohere Labs Command R+ model

class RAG:
    """
    A class for Retrieval-Augmented Generation (RAG) system.
    It handles retrieving relevant passages from an index based on a query
    and then generating a response using a large language model (LLM)
    conditioned on those passages.
    """

    def __init__(self, index_name="/home/alan.dryaev/msmarco-v2-passage",
                llm_name="CohereLabs/c4ai-command-r-plus-4bit",
                top_k=5):
        """
        Initializes RAG components.
        
        Args:
            index_name (str): path to index, default: MSMARCO v2 passage
            llm_name (str): identifier of language model, default: Cohere Labs Command R+ 
            top_k (int): maximum number of passages retrieved from index
        """
        self.index_name = index_name
        self.llm_name = llm_name
        self.top_k = top_k

        # Initialize components to None and then create the objects
        self.retriever = None
        self.tokenizer = None
        self.generator = None
        self.initialize_retriever()
        self.initialize_llm()


    def initialize_retriever(self):
        """      
        Initialize Pyserini Searcher for passage retrieval
        """
        print(f"Initializing Retriever...")
        try:
            self.retriever = LuceneSearcher(self.index_name)
            #self.retriever = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")
            print("Retriever initialized successfully.")

        except Exception as e:
            print(f"Error initializing Retriever: {e}")
            raise

    def initialize_llm(self):
        """      
        Initialize tokenizer and LLM from Cohere Labs Command R+
        """
        try: 
            print(f"Initializing LLM...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name)

            self.generator = AutoModelForCausalLM.from_pretrained(
                self.llm_name,
                device_map="auto"
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            raise
    
    def retrieve(self, query):
        """
        Retrieves passages from the index based on the query.
        
        Args:
            query (str): users question

        Returns:
            list[dict]: list of retrieved passages, each formatted as a dictionary.
        """
        retrieved_passages = []
        #get top k passages for query
        results = self.retriever.search(query, k=self.top_k)

        #format each passage
        for id,hit in enumerate(results):

            #get raw information of passage and extract the content (docid in this case is equivalent to passage id)
            pas_id = hit.docid
            pas_raw = self.retriever.doc(pas_id).raw()
            pas_json = json.loads(pas_raw)
            pas_content = pas_json.get('passage', pas_raw)
            score = hit.score

            # brings the document in a format of a dictionary expected by the generator
            retrieved_passages.append({
                # Information for the generator
                "title": f"Document {id}",
                "text": pas_content,
                # Metadata
                "id": id,
                "pas_id": pas_id,
            })
        return retrieved_passages

    def generate(self, query, retrieved_passages):
        """
        Generates an answer using the LLM based on the query and retrieved passages.
        
        Args:
            query (str): users question
            retrieved_passages (list[dict]): The list of passages retrieved from index

        Returns:
            str: generated answer from the LLM.
        """
        if not retrieved_passages:
            return "No relevant passages found."
        # Prompt structure optimized for Document citation

        docs_for_llm = [{"title": pas["title"], "text": pas["text"]} for pas in retrieved_passages]
        message = [{
            "role": "user", 
            "content": f"{query}"
            }]

        prompt = self.tokenizer.apply_grounded_generation_template(
            message,
            documents=docs_for_llm,
            citation_mode="accurate",
            tokenize=True,  #tokenize later
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.generator.device)
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).input_ids.to(self.generator.device)
        """    
        # Generate answer tokens from input with LLM
        gen_tokens = self.generator.generate(
            prompt, 
            max_new_tokens=300, 
            do_sample=True, 
            temperature=0.3,
        )

        # Slice input tokens to get answer tokens
        answer_tokens = gen_tokens[0][prompt.shape[-1]:]

        # Decodes generated answer into a string and removes special LLM tokens
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer.strip()

def run(rag, dataset, result_file, progress_file):

    # Batch size is relatively small, duplicates are deleted manually
    BATCH_SIZE = 25
    N_LOOPS = 300
    batch_nr = 0
    # Check progress in progress file
    try:
        with open(progress_file, "r") as f:
            start_index = json.load(f).get("next_item_index", 0)
        print(f"Resuming from index {start_index}.")
    except (FileNotFoundError, json.JSONDecodeError):
        start_index = 0
        print("No Process File.")
    batch_nr = int(start_index / BATCH_SIZE) + 1

    with open(result_file, "a") as results_f:
        for n in range(0, N_LOOPS):
            items = dataset.select(range(start_index, start_index + BATCH_SIZE))
            for i,item in enumerate(items):
                query = item.get("text")
                if not query:
                    continue
                query_id = item['_id']
                print(f"Processing item {i}: {query[:60]}...")

                # RAG pipeline: retrieve docuemnts, get answer from LLM
                retrieved_passages = rag.retrieve(query)
                answer = rag.generate(query, retrieved_passages)
                retrieved_pas_ids = [pas["pas_id"] for pas in retrieved_passages]


                result_data = {
                    "query_id": query_id,
                    "query": query,
                    "answer": answer,
                    "retrieved_pas_ids": retrieved_pas_ids,

                }

                json_string = json.dumps(result_data)
                # Write the JSON string as a new line in the .jsonl file
                results_f.write(json_string + "\n")
                results_f.flush()

            start_index += BATCH_SIZE
            with open(PROGRESS_FILE, "w") as progress_f:
                json.dump({"next_item_index": start_index}, progress_f)
            print(f"--- Progress checkpoint saved. Next run will start at {start_index}. ---")
            batch_nr+=1

            print(f"BATCH {batch_nr} complete.")


    
if __name__ == "__main__":
    print("Starting RAG initialization")

    # Initialize RAG setup
    try:
        rag = RAG()
    except Exception as e:
        print(f"Could not initialize the RAG system: {e}")
        exit()

    # Test retrieve function
    print("finish RAG initialization")
    print("test RAG function")

    #test evaluation function
    # from your_evaluator_module import evaluator
    # from your_rag_module import rag

    # Results: LLM Answer
    # Progress: number of last batch is saved
    RESULTS_FILE = "results.jsonl" 
    PROGRESS_FILE = "progress.json"


    full_dataset = load_dataset("mteb/msmarco-v2", "queries", split="queries")
    results = run(rag, full_dataset, RESULTS_FILE, PROGRESS_FILE)


