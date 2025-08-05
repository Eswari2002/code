import os
import json
import pdfplumber
import pandas as pd
import camelot
import numpy as np
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from google import genai
from google.genai.types import EmbedContentConfig

# API Setup
GEMINI_API_KEY = "AIzaSyDqAYD6aPGt9FNGI55rBFQDqPdcjwTNnCg" #"AIzaSyDg4-xN4ItnZ0d6gyu3rQWpgt8Pt0MogNw"
client = genai.Client(api_key=GEMINI_API_KEY)


# === Parameters ===
target_header = "consolidated balance sheet"  # Semantic intent
similarity_threshold = 0.75  # Adjust based on experimentation

top_k = 3
top_k1 = 5

def clean_chunks(chunks):
    return [chunk.strip() for chunk in chunks if isinstance(chunk, str) and chunk.strip()]

def process_excel_to_json(excel_file_path):
    try:
        excel_data = pd.ExcelFile(excel_file_path)
        json_data = {}
        for sheet_name in excel_data.sheet_names:
            df = excel_data.parse(sheet_name)
            json_data[sheet_name] = df.to_dict(orient="records")
        return json_data
    except Exception as e:
        print(f"Error processing Excel file '{excel_file_path}': {e}")
        return {}

def extract_notes_mapping_from_pdf(pdf_path):
    notes, current_note, current_text = {}, None, ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text: continue
            for line in text.splitlines():
                match = re.match(r"Note\s+(\d+)[\s:\-]", line.strip(), re.IGNORECASE)
                if match:
                    if current_note:
                        notes[current_note] = {"page": page.page_number, "content": current_text.strip()}
                    current_note = f"Note {match.group(1)}"
                    current_text = line
                elif current_note:
                    current_text += " " + line
        if current_note:
            notes[current_note] = {"page": len(pdf.pages), "content": current_text.strip()}
    return notes

def generate_google_embeddings(contents):
    if not contents or not isinstance(contents, list) or all(not content.strip() for content in contents):
        raise ValueError("Contents must be a non-empty list of non-empty strings.")
    try:
        embeddings = []
        for i in range(0, len(contents), 100):
            batch = contents[i:i + 100]
            response = client.models.embed_content(
                model="text-embedding-004",
                contents=batch,
                config=EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
            )
            embeddings.extend([emb.values for emb in response.embeddings])
        return embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []

def extract_page_header_from_chunk(chunk):
    """
    Extract the 'Page Header' from metadata in the chunk.
    Example metadata line: 'Page Header: Statement of Financial Position | Group Report'
    """
    match = re.search(r"Page Header:\s*(.+)", chunk, re.IGNORECASE)
    return match.group(1).strip() if match else None

def extract_table_header_from_chunk(chunk):
    """
    Extract the 'Table Header' from metadata in the chunk.
    Example line: 'Table Header: Consolidated Balance Sheet'
    """
    match = re.search(r"Table Header:\s*(.+)", chunk, re.IGNORECASE)
    return match.group(1).strip() if match else None


def search_google_embeddings(query, chunks, chunk_embeddings, top_k, filter_header=None):
    chunks = clean_chunks(chunks)
    if not chunks:
        print(f"No valid chunks to search for query: {query}")
        return []

    try:
        chunk_embeddings = np.array(chunk_embeddings)

        # If a filter_header is provided, try exact match first
        if filter_header:
            exact_filtered_chunks = []
            exact_filtered_embeddings = []

            for i, chunk in enumerate(chunks):
                page_header = extract_page_header_from_chunk(chunk)
                if page_header and page_header.strip().lower() == filter_header.strip().lower():
                    exact_filtered_chunks.append(chunk)
                    exact_filtered_embeddings.append(chunk_embeddings[i])

            if exact_filtered_chunks:
                chunks = exact_filtered_chunks
                chunk_embeddings = np.array(exact_filtered_embeddings)
            else:
                # Fall back to semantic filtering if no exact matches
                filter_embedding = generate_google_embeddings([filter_header])[0]
                semantic_filtered_chunks = []
                semantic_filtered_embeddings = []

                for i, chunk in enumerate(chunks):
                    table_header = extract_table_header_from_chunk(chunk)
                    if table_header:
                        header_embedding = generate_google_embeddings([table_header])[0]
                        similarity = float(np.dot(filter_embedding, header_embedding))

                        if similarity >= 0.75:  # Threshold for semantic match
                            semantic_filtered_chunks.append(chunk)
                            semantic_filtered_embeddings.append(chunk_embeddings[i])

                chunks = semantic_filtered_chunks
                chunk_embeddings = np.array(semantic_filtered_embeddings)

                if not chunks:
                    print(f"No matching chunks found for header (exact or semantic): {filter_header}")
                    return []

        # Generate embedding for the query
        query_embedding = generate_google_embeddings([query])[0]

        # Compute similarity
        similarities = np.dot(chunk_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [{"chunk": chunks[i], "score": float(similarities[i])} for i in top_indices]

    except Exception as e:
        print(f"Error during embedding search for query '{query}': {e}")
        return []

def rank_and_format_chunks(chunks_b, chunks_c, top_n=4):
    combined = chunks_b + chunks_c
    combined_sorted = sorted(combined, key=lambda x: x["score"], reverse=True)
    top_chunks = combined_sorted[:top_n]
    # Optional: format chunks nicely for display
    return "\n\n---\n\n".join([f"Score: {round(chunk['score'], 4)}\n{chunk['chunk']}" for chunk in top_chunks])

def extract_pdf_chunks(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = "**".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=40)
    return clean_chunks(splitter.split_text(full_text))

def main():
    print("=== PDF Embed & Retrieve Questions from Excel (B & C Columns) ===")
    pdf_path = r"C:\Projects\FinPopulationNotes\volex.pdf"
    query_excel_path = r"C:\Projects\FinPopulationNotes\output_with_header.xlsx"
    output_csv_path = "retrieved_chunk_filter_volex_direct.csv"

    base_name = Path(pdf_path).stem
    os.makedirs("embedding_volex_new", exist_ok=True)
    EMBEDDINGS_CACHE = f"embedding_volex_new/{base_name}_embedded_text.json"
    EXCEL_FILE_STREAM = f"embedding_volex_new/{base_name}_stream.xlsx"
    EXCEL_FILE_LATTICE = f"embedding_volex_new/{base_name}_lattice.xlsx"
    JSON_FILE_STREAM = f"embedding_volex_new/{base_name}_stream.json"
    JSON_FILE_LATTICE = f"embedding_volex_new/{base_name}_lattice.json"
    NOTES_EXCEL_PATH = f"embedding_volex_new/{base_name}_AllNotes.xlsx"
    NOTE_EMBEDDINGS_CACHE = f"embedding_volex_new/{base_name}_note_embeddings.json"

    notes_map = extract_notes_mapping_from_pdf(pdf_path)

    if not os.path.exists(JSON_FILE_STREAM) or not os.path.exists(JSON_FILE_LATTICE):
        print("Processing PDF and extracting tables...")

        # Extract page-level text for page headers using pdfplumber
        with pdfplumber.open(pdf_path) as pdf_plumber:
            pdf_pages = [page.extract_text() for page in pdf_plumber.pages]
            notes_map = extract_notes_mapping_from_pdf(pdf_path)
            
        with pd.ExcelWriter(EXCEL_FILE_STREAM, engine='xlsxwriter') as writer:
            stream_tables = camelot.read_pdf(pdf_path, flavor="stream", pages='all')
            for i, table in enumerate(stream_tables):
                page_num = table.page
                table_index = i + 1
                table_df = table.df
                
                # Extract table header
                table_header = ', '.join(table_df.iloc[0].dropna().tolist()) if not table_df.empty else "N/A"

                # Get page-level header from pdfplumber
                page_text = pdf_pages[page_num - 1] if page_num <= len(pdf_pages) else ""
                page_lines = page_text.splitlines() if page_text else []
                page_header = " | ".join(page_lines[:3]) if page_lines else "N/A"
                notes_found = re.findall(r"Note\s+\d+", table_df.to_string())
                notes_texts = [notes_map.get(n, "") for n in notes_found]
                
                # Metadata rows
                metadata = pd.DataFrame({
                    "Metadata": [
                        f"Page: {page_num}",
                        f"Table Index: {table_index}",
                        f"Table Header: {table_header}",
                        f"Page Header: {page_header}",
                        f"Notes Mentioned: {', '.join(notes_found) or 'None'}"

                    ]
                })
                
                #Combine metadata and table
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Stream_Table_{i+1}", index=False, header=False)

        #Process Lattice tables
        with pd.ExcelWriter(EXCEL_FILE_LATTICE, engine='xlsxwriter') as writer:
            lattice_tables = camelot.read_pdf(pdf_path, flavor="lattice", pages='all')
            for i, table in enumerate(lattice_tables):
                page_num = table.page
                table_index = i + 1
                table_df = table.df
                
                # Extract table header
                table_header = ', '.join(table_df.iloc[0].dropna().tolist()) if not table_df.empty else "N/A"

                # Get page-level header from pdfplumber
                page_text = pdf_pages[page_num - 1] if page_num <= len(pdf_pages) else ""
                page_lines = page_text.splitlines() if page_text else []
                page_header = " | ".join(page_lines[:3]) if page_lines else "N/A"
                notes_found = re.findall(r"Note\s+\d+", table_df.to_string())
                notes_texts = [notes_map.get(n, "") for n in notes_found]
                
                # Metadata rows
                metadata = pd.DataFrame({
                    "Metadata": [
                        f"Page: {page_num}",
                        f"Table Index: {table_index}",
                        f"Table Header: {table_header}",
                        f"Page Header: {page_header}",
                        f"Notes Mentioned: {', '.join(notes_found) or 'None'}"

                    ]
                })

                # Combine metadata and table
                new_table = pd.concat([metadata, table_df], ignore_index=True)
                new_table.to_excel(writer, sheet_name=f"Lattice_Table_{i+1}", index=False, header=False)

        json.dump(process_excel_to_json(EXCEL_FILE_STREAM), open(JSON_FILE_STREAM, 'w', encoding='utf-8'), indent=4)
        json.dump(process_excel_to_json(EXCEL_FILE_LATTICE), open(JSON_FILE_LATTICE, 'w', encoding='utf-8'), indent=4)

    if os.path.exists(EMBEDDINGS_CACHE):
        with open(EMBEDDINGS_CACHE, 'r', encoding='utf-8') as f:
            cached_embeddings = json.load(f)
            table_chunks1 = cached_embeddings.get("stream_tables", [])
            table_chunks2 = cached_embeddings.get("lattice_tables", [])
            docs = cached_embeddings.get("docs", [])
    else:
        table_chunks1 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_STREAM)).values()])
        table_chunks2 = clean_chunks([json.dumps(table) for table in json.load(open(JSON_FILE_LATTICE)).values()])
        with pdfplumber.open(pdf_path) as pdf:
            document_texts = "**".join([page.extract_text() for page in pdf.pages if page.extract_text()])
        docs = clean_chunks(RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=40).split_text(document_texts))
        with open(EMBEDDINGS_CACHE, 'w', encoding='utf-8') as f:
            json.dump({"stream_tables": table_chunks1, "lattice_tables": table_chunks2, "docs": docs}, f, indent=4)

    # Combine all chunks
    chunks = docs + table_chunks1 + table_chunks2
    embeddings = generate_google_embeddings(chunks)

    if os.path.exists(NOTE_EMBEDDINGS_CACHE):
        with open(NOTE_EMBEDDINGS_CACHE, 'r', encoding='utf-8') as f:
            note_embeddings = json.load(f)
    else:
        note_embeddings = dict(zip(
            notes_map.keys(),
            [{'embedding': emb, 'page': notes_map[k]['page']} for k, emb in zip(notes_map.keys(), generate_google_embeddings([notes_map[k]['content'] for k in notes_map.keys()]))]
        ))
        with open(NOTE_EMBEDDINGS_CACHE, 'w', encoding='utf-8') as f:
            json.dump(note_embeddings, f, indent=4)

    with pd.ExcelWriter(NOTES_EXCEL_PATH, engine="xlsxwriter") as writer:
        for note_id, note_data in notes_map.items():
            df = pd.DataFrame({
                "Header": [note_id],
                "Page": [note_data["page"]],
                "Content": [note_data["content"]]
            })
            sheet_name = note_id.replace(" ", "_")[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)

        embed_rows = []
        for nid, data in note_embeddings.items():
            embed_rows.append({
                "Note ID": nid,
                "Page": data['page'],
                "Embedding": json.dumps(data['embedding'])[:500]
            })
        embed_df = pd.DataFrame(embed_rows)
        embed_df.to_excel(writer, sheet_name="Embeddings_Index", index=False)

    print("Notes extraction and embedding complete. Excel and embedding files saved.")
    
    # Process Questions from Excel
    df = pd.read_excel(query_excel_path, sheet_name="output_with_header")
    results = []
    for index, row in df.iterrows():
        id_value = row[0]
        question_b = str(row[1]).strip() if len(row) > 1 and str(row[1]).strip().lower() != "nan" else ""
        question_c = str(row[2]).strip() if len(row) > 2 and str(row[2]).strip().lower() != "nan" else ""

        # chunks_b = search_google_embeddings(question_b, chunks, embeddings, top_k1) if question_b else []
        # chunks_c = search_google_embeddings(question_c, chunks, embeddings, top_k1) if question_c else []

        # filter_header = str(row.get('header', '')).strip()
        filter_header = target_header

        chunks_b = search_google_embeddings(question_b, chunks, embeddings, top_k1, filter_header=filter_header) if question_b else []
        chunks_c = search_google_embeddings(question_c, chunks, embeddings, top_k1, filter_header=filter_header) if question_c else []

        formatted_b = "\n---\n".join([c["chunk"] for c in chunks_b])
        formatted_c = "\n---\n".join([c["chunk"] for c in chunks_c])

        ranked_combined = rank_and_format_chunks(chunks_b, chunks_c, top_n=4)

        # Extract notes from Retrieved B
        note_ids_b = re.findall(r"Note\s+\d+", formatted_b)
        notes_b = []
        if note_ids_b:
            try:
                query_embedding_b = generate_google_embeddings([question_b])[0]
                for note_id in set(note_ids_b):
                    if note_id in note_embeddings:
                        note_vector = np.array(note_embeddings[note_id]["embedding"])
                        score = float(np.dot(query_embedding_b, note_vector))
                        notes_b.append((note_id, notes_map[note_id]["content"], score))
                notes_b.sort(key=lambda x: x[2], reverse=True)
                top_notes_b = "\n\n---\n\n".join([f"{n[0]} (Score: {round(n[2], 4)}):\n{n[1]}" for n in notes_b[:2]])
            except Exception as e:
                top_notes_b = f"Error matching notes B: {e}"
        else:
            top_notes_b = "No relevant notes found in Retrieved B."

        # Extract notes from Retrieved C
        note_ids_c = re.findall(r"Note\s+\d+", formatted_c)
        notes_c = []
        if note_ids_c:
            try:
                query_embedding_c = generate_google_embeddings([question_c])[0]
                for note_id in set(note_ids_c):
                    if note_id in note_embeddings:
                        note_vector = np.array(note_embeddings[note_id]["embedding"])
                        score = float(np.dot(query_embedding_c, note_vector))
                        notes_c.append((note_id, notes_map[note_id]["content"], score))
                notes_c.sort(key=lambda x: x[2], reverse=True)
                top_notes_c = "\n\n---\n\n".join([f"{n[0]} (Score: {round(n[2], 4)}):\n{n[1]}" for n in notes_c[:2]])
            except Exception as e:
                top_notes_c = f"Error matching notes C: {e}"
        else:
            top_notes_c = "No relevant notes found in Retrieved C."

        
        # NEW LOGIC: Extract notes from retrieved chunks and match using note embeddings
        all_text = formatted_b + " " + formatted_c
        note_ids_in_text = re.findall(r"Note\s+\d+", all_text)

        note_results = []
        if note_ids_in_text:
            try:
                combined_query = (question_b + " " + question_c).strip()
                query_embedding = generate_google_embeddings([combined_query])[0]

                for note_id in set(note_ids_in_text):
                    if note_id in note_embeddings:
                        note_vector = np.array(note_embeddings[note_id]["embedding"])
                        score = float(np.dot(query_embedding, note_vector))
                        note_results.append((note_id, notes_map[note_id]["content"], score))

                note_results.sort(key=lambda x: x[2], reverse=True)
                top_notes = "\n\n---\n\n".join([f"{n[0]} (Score: {round(n[2], 4)}):\n{n[1]}" for n in note_results[:2]])
            except Exception as e:
                top_notes = f"Error matching notes: {e}"
        else:
            top_notes = "No relevant note references found."
            
        results.append({
            "Column A": id_value,
            "Question B": question_b,
            "Retrieved B": formatted_b,
            "Relevant Notes B": top_notes_b,
            "Question C": question_c,
            "Retrieved C": formatted_c,
            "Relevant Notes C": top_notes_c,
            "Top Ranked Chunks (B+C)": ranked_combined,
            "Top Related Notes": top_notes
        })

    # for index, row in df.iterrows():
    #     id_value = row[0]
    #     question_b = str(row[1]).strip() if len(row) > 1 and str(row[1]).strip().lower() != "nan" else ""
    #     question_c = str(row[2]).strip() if len(row) > 2 and str(row[2]).strip().lower() != "nan" else ""

    #     retrieved_b = "\n---\n".join(search_google_embeddings(question_b, chunks, embeddings, top_k)) if question_b else ""
    #     retrieved_c = "\n---\n".join(search_google_embeddings(question_c, chunks, embeddings, top_k)) if question_c else ""

    #     results.append({
    #         "Column A": id_value,
    #         "Question B": question_b,
    #         "Retrieved B": retrieved_b,
    #         "Question C": question_c,
    #         "Retrieved C": retrieved_c
    #     })
        
    output_df = pd.DataFrame(results)
    output_df.to_csv(output_csv_path, index=False)
    print(f"\n✅ Retrieved chunks and notes saved to: {output_csv_path}")

if __name__ == "__main__":
    main()
