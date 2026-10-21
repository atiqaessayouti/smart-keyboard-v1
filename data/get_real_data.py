import nltk
import csv
import os

print("Downloading real-world text data...")
nltk.download('brown', quiet=True)
from nltk.corpus import brown

def generate_real_corpus():
    os.makedirs("data", exist_ok=True)
    path = "data/corpus.csv"
    
    with open(path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["doc_id", "sentence"]) 
        
        doc_id = 0
        sentence_count = 0
        
        for fileid in brown.fileids():
            if doc_id >= 50: 
                break
                
            # استعملنا brown.sents() باش نحيّدو الـ POS Tags
            sentences = brown.sents(fileid)
            
            for sent in sentences:
                # نربطو الكلمات بلا Tags ونحولوهم لحروف صغيرة
                clean_sent = " ".join([w.lower() for w in sent])
                
                if len(clean_sent.split()) > 5:
                    writer.writerow([doc_id, clean_sent])
                    sentence_count += 1
            doc_id += 1
            
    print(f"✅ Generated clean real corpus at {path} ({sentence_count} sentences, {doc_id} docs)")

if __name__ == "__main__":
    generate_real_corpus()