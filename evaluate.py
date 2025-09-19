import rag
import json
from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import IndexReader
from collections import Counter, defaultdict
import re
from urllib.parse import urlparse
import csv


def weight_dummy(document_id):
    """default weighting function that returns 1 for any document id"""
    return 1


def count_domains(result_file, doc_retriever, pas_retriever, weighting_function=weight_dummy):
    """
    Processes a results file to count the domains of cited documents.

    This function reads a JSONL file containing generated answers and their
    retrieved passages. It parses custom citation tags (e.g., "<co: 1,2>text</co: 1,2>")
    from the answers to identify which passages were cited. For each cited passage,
    it retrieves the full document, extracts its source URL, and parses the
    domain. It then counts the occurrences of each domain, applying a
    customizable weight to each count. 

    The function produces two types of counts:
    1.  A "single" count: A domain is counted once per cited document in an answer.
    2.  A "multiple" count: A domain is counted for every single citation instance
        it receives in an answer.

    Args:
        result_file (str): The path to the input JSONL file containing the generation
                           results. Each line should be a JSON object with "answer"
                           and "retrieved_docs" keys.
        doc_retriever: An object to retrieve raw content of a full document by its ID.
        pas_retriever: An object to retrieve raw content of a passage by its ID.
        weighting_function (callable, optional): A function that accepts a document ID
                                                 and returns a numerical weight (float or int).
                                                 Defaults to `weight_dummy`, which returns 1.

    Returns:
        tuple[Counter, Counter]: A tuple containing two Counter objects:
                                 - domain_counter_single: Counts domains once per cited document.
                                 - domain_counter_multiple: Counts domains for each citation instance.
    """
    domain_counter_single = Counter()
    domain_counter_multiple = Counter()

    # Use a defaultdict with a LIST to store every citation instance
    domain_docids_single = defaultdict(list)
    domain_docids_multiple = defaultdict(list)



    with open(result_file, "r") as results_f:

        # Process each line in the results file.
        for line in results_f:
            data = json.loads(line)
            answer = data.get("answer")
            retrieved_pas_ids = data.get("retrieved_pas_ids")

            # This captures the numeric indices (e.g., "1,2") between the tags.
            matches = re.findall(r"<co: ([\d,]+)>(.*?)<\/co: \1>", answer)
            cited_indices = []

            # Strings of document indices formatted to integer list
            for match in matches:
                documents = match[0]
                indices = documents.split(',')
                for doc in indices:
                    cited_indices.append(int(doc))

            # Iterate through passages that were retrieved for this answer.
            for index, pas_id in enumerate(retrieved_pas_ids):
                if index in cited_indices:

                        # Extract passage content
                        pas_raw = pas_retriever.doc(pas_id).raw()
                        pas_json = json.loads(pas_raw)

                        # Get document id from passage content
                        doc_id = pas_json.get('docid', pas_raw)

                        # Extract document content
                        document = None
                        try:
                            document = doc_retriever.doc(doc_id).raw()
                            doc_json = json.loads(document)
                        except KeyError:
                            print(f"Document with ID '{doc_id}' not found.")

                        
                        # Count the domains for every occurence of an indice single / mulitple time
                        if document is not None:
                            url = doc_json.get('url', 'no_url')
                            if url != 'no_url':
                                # Parse the URL and get the domain (netloc)
                                parsed_url = urlparse(url)
                                domain = parsed_url.netloc

                                # Lookup weight for document id
                                """calculate weight for document. weighting function is passed as parameter"""
                                doc_weight = weighting_function(doc_id)
                                
                                # Increase the Counters for every doc id correspondently
                                domain_counter_single[domain]+=1*doc_weight
                                count = cited_indices.count(index)
                                domain_counter_multiple[domain]+=count*doc_weight

                                # Append the doc_id to the lists correspondently
                                for i in range(count):
                                    domain_docids_multiple[domain].append(doc_id)

                                domain_docids_single[domain].append(doc_id)

    return domain_counter_single, domain_counter_multiple, domain_docids_single, domain_docids_multiple



if __name__ == "__main__":
    RESULTS_FILE = "results.jsonl"
    doc_retriever = LuceneSearcher("/home/alan.dryaev/msmarco-v2-doc")
    #doc_retriever = LuceneSearcher.from_prebuilt_index("msmarco-v2-doc")
    #pas_retriever = LuceneSearcher.from_prebuilt_index("msmarco-v2-passage")
    pas_retriever = IndexReader("/home/alan.dryaev/msmarco-v2-passage")

    #dcs single count, dcm multiple count, dtd domain to docs
    dcs, dcm, dtds, dtdm = count_domains(RESULTS_FILE, doc_retriever, pas_retriever)

    dcs = dcs.most_common()
    dcm = dcm.most_common()
    
    #save evaluation results in the files dcs_counts, dcm_counts, domain_docids_single, domain_docids_multiple
    with open('dcs_counts.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Website', 'Count'])
        writer.writerows(dcs)

    with open('dcm_counts.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Website', 'Count'])
        writer.writerows(dcm)

    with open('domain_docids_single.jsonl', "w") as f_out:
        json.dump(dtds, f_out, indent=4)
    with open('domain_docids_multiple.jsonl', "w") as f_out:
        json.dump(dtdm, f_out, indent=4)