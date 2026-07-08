import os
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class Neo4jGraph:
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI")
        self.user = os.getenv("NEO4J_USERNAME")
        self.password = os.getenv("NEO4J_PASSWORD")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Test connection
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j Graph Database!")
        except Exception as e:
            print(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def add_equipment(self, equipment_tag, equipment_type="Equipment"):
        """Adds an equipment node to the graph"""
        if not self.driver:
            return
        query = (
            "MERGE (e:Equipment {tag: $tag}) "
            "SET e.type = $type "
            "RETURN e"
        )
        with self.driver.session() as session:
            session.run(query, tag=equipment_tag, type=equipment_type)

    def add_connection(self, source_tag, target_tag, relationship_type="CONNECTED_TO"):
        """Creates a relationship between two equipment nodes"""
        if not self.driver:
            return
        query = (
            "MERGE (s:Equipment {tag: $source}) "
            "MERGE (t:Equipment {tag: $target}) "
            f"MERGE (s)-[:{relationship_type}]->(t)"
        )
        with self.driver.session() as session:
            session.run(query, source=source_tag, target=target_tag)
            
    def add_document_reference(self, doc_name, equipment_tag):
        """Links a document (like an SOP) to an equipment tag"""
        if not self.driver:
            return
        query = (
            "MERGE (d:Document {name: $doc_name}) "
            "MERGE (e:Equipment {tag: $tag}) "
            "MERGE (d)-[:MENTIONS]->(e)"
        )
        with self.driver.session() as session:
            session.run(query, doc_name=doc_name, tag=equipment_tag)

neo4j_graph = Neo4jGraph()
