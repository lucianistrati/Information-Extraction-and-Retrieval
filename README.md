# Information Extraction and Retrieval

## Project Description

**Information Extraction and Retrieval** is a Python-based project that leverages NLP techniques for extracting meaningful entities, relationships, and creating knowledge graphs from unstructured text. The project uses Spark NLP for named entity recognition and Spacy for sentence processing, enabling efficient information extraction from text documents.

## Key Features

- Named Entity Recognition (NER) for PERSON and LOCATION using Spark NLP
- Extraction of subject-object-relation triples from sentences
- Visualization of relationships in a knowledge graph using NetworkX and Matplotlib
- RDF graph creation and querying using the rdflib library

## Requirements

- Python 3.x
- Java 8 (required for Spark)
- Apache Spark with Spark NLP
- Spacy
- RDFlib
- NetworkX
- Matplotlib

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/information-extraction-and-retrieval.git
   cd information-extraction-and-retrieval
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Required Packages**:
   You can install the required packages using `pip`:
   ```bash
   pip install pyspark==3.2.1
   pip install spark-nlp==3.3.1
   pip install spacy==3.2.0
   pip install rdflib==6.1.1
   pip install networkx==2.5
   pip install matplotlib==3.5.0
   ```

4. **Download Spacy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## Usage

### Running the Project

To run the main extraction and retrieval process, execute the following command:

```bash
python src/main.py
```

### Key Components

#### 1. `main.py`
- Responsible for initializing the Spark session and performing named entity recognition.
- Reads texts, extracts PERSON and LOCATION entities, and outputs the results.

#### 2. `knowledge_graph.py`
- Processes sentences to extract subject-object-relation triples.
- Creates an RDF graph and visualizes relationships using NetworkX and Matplotlib.
- Includes functions for sentence segmentation and token processing.

### Sample Output

The program extracts named entities (e.g., persons and locations) and prints them, along with generating a knowledge graph of relationships. 

## Data Files

The project uses text files for entity matching, which are saved in the `data` directory:

- `data/person_matches.txt` – Contains names of persons to be matched.
- `data/location_matches.txt` – Contains names of locations to be matched.

## Contributing

Contributions are welcome! If you would like to contribute to this project, please fork the repository and create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Spark NLP](https://nlp.johnsnowlabs.com/) for providing a powerful NLP library.
- [Spacy](https://spacy.io/) for natural language processing tasks.
- [NetworkX](https://networkx.org/) and [Matplotlib](https://matplotlib.org/) for graph visualization.
- [RDFlib](https://rdflib.readthedocs.io/) for working with RDF in Python.
```
