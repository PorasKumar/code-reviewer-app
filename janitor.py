from pinecone import Pinecone
import time
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not API_KEY or not INDEX_NAME:
    print("Environment Variables missing for Janitor sweep ❌")
    exit(1)

pc = Pinecone(api_key=API_KEY)
index = pc.Index(INDEX_NAME)

#fetch namespaces
try:
    stats = index.describe_index_stats()
    namespaces = stats.get("namespaces",{})
except Exception as e:
    print(f"❌ Failed to fetch index stats from Pinecone: {e}")
    exit(1)

current_time = (time.time())
MAX_AGE_SECONDS = 10800

print(f"\n🧹 Running Janitor Sweep at {time.ctime()}...")
print(f"Found {len(namespaces)} active namespaces.")

for nmspc in list(namespaces.keys()):
    try:
        # Expecting format user_ID(int)_TIMESTAMP(int)
        parts = nmspc.split("_")
        timestamp_str = parts[-1]
        nmspc_timestamp = float(timestamp_str)

        #checking if age is more than MAX_AGE_SECONDS
        if(current_time - nmspc_timestamp) >= MAX_AGE_SECONDS:
            print(f"🗑️ Deleting expired namespace: {nmspc}")

            # Compatible with all Pinecone SDK versions
            try:
                index.delete(delete_all=True, namespace=nmspc)
            except AttributeError:
                # Fallback for newer SDK versions
                index.delete_all(namespace=nmspc)

        else:
            print(f"⏳ Keeping Active namespace: {nmspc}")

    except (ValueError, IndexError):
            # Skip namespaces that don't match our timestamp format (e.g., default namespace)
            print(f"⚠️ Skipping non-standard namespace: {nmspc}")
    except Exception as e:
        print(f"❌ Failed to delete namespace '{nmspc}'\nErro: {e}")

print("✨ Janitor sweep completed successfully!")