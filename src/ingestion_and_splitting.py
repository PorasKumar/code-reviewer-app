from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
from typing import List
import os
import io
import re
import requests
import zipfile
from dotenv import load_dotenv
load_dotenv()

#The splitter will look for syntax of these languages to split smartly
EXTENSION_MAP = {
    ".py": Language.PYTHON,
    ".js": Language.JS,
    ".ts": Language.TS,
    ".jsx": Language.JS,
    ".tsx": Language.TS,
    ".java": Language.JAVA,
    ".cpp": Language.CPP,
    ".c": Language.C,
    ".cs": Language.CSHARP,
    ".go": Language.GO,
    ".rb": Language.RUBY,
    ".rs": Language.RUST,
    ".php": Language.PHP,
    ".html": Language.HTML,
    ".sol": Language.SOL,
    ".md": Language.MARKDOWN,
}

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

#files that we have to ignore, we only need to split human written code
IGNORED_DIRS = {".git",".gitignore", "__pycache__", "node_modules", "venv", ".venv", "build", "dist"}
IGNORED_EXTENSIONS = {".lock",".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".exe", ".pyc", ".zip", ".tar",".db",".sqlite3"}
IGNORED_EXACT_FILES = {"uv.lock", "package-lock.json", "yarn.lock", "poetry.lock", "Pipfile.lock", "Cargo.lock"}


def process_repo_in_memory(pr_url:str)->List[Document]:
    """Extracts owner, reponame, branch using the pr_url for downloading repository as zip directly into RAM. 
    Read the file without writing to disk, then we split into chunks and convert into List[Document] for pinecone upserting"""

    try:
        #################################################################
        #extract owner, reponame, pull number, branch_name using  pr_url#
        #################################################################

        pattern = r"github\.com/([^/]+)/([^/]+)/pull/(\d+)"
        match = re.search(pattern, pr_url)

        if not match:
            raise ValueError("Invalid GitHub PR URL. Example format: https://github.com/owner/repo/pull/12")

        owner, repo, pull_number = match.groups()

        #calling github api to get target branch(base.ref) (because we need branch name to download zip file of repo)
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls/{pull_number}"
        headers = {"Accept": "application/vnd.github.v3+json"}

        if GITHUB_TOKEN:
            headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

        res = requests.get(api_url, headers=headers)
        if res.status_code !=200:
            raise RuntimeError("Failed to fetch PR details for branch name in ingestion phase")

        #extract it from json
        pr_data = res.json()
        #Use head['sha'] (or head['ref']) to target the PR branch instead of 'main'
        #if we use ['sha'] then it will it retrieve the exact state of the repository at that precise commit, frozen in time.
        pr_branch_name = pr_data["head"]["sha"]


        #########################################################
        #download the repository in RAM and make List[Documents]#
        #########################################################

        #github api archive(zip) ke liye url
        archive_url = f"https://api.github.com/repos/{owner}/{repo}/zipball/{pr_branch_name}"


        #authorization of GITHUB_TOKEN (already done above)
        print(f"Downloading repository archive for {owner}/{repo} Branch:({pr_branch_name}) into RAM...")
        response = requests.get(archive_url, headers=headers, stream=True)


        if response.status_code !=200:
            raise RuntimeError(f"Failed to download archive zip of repository into RAM: Status Code{response.status_code}")

        #we'll store the repo files' text chunks in this
        all_chunks : List[Document] = []

        #opening the raw byte stream in memory
        zip_buffer = io.BytesIO(response.content)

        #open the zipfile
        with zipfile.ZipFile(zip_buffer) as zf:
            #iterate each file/folder in the zip file
            for zip_info in zf.infolist():

                #skip directories/folders   
                #it'll not skip the code files, When a ZIP archive is created, it contains separate entries for folders and files 
                if zip_info.is_dir():
                    continue

                #Below we extract relative path by splitting the top level github folder name
                #"owner-repo-commit_sha/src/main.py"  -->  "src/main.py" is what we'll have in path_parts[1]
                path_parts = zip_info.filename.split("/",1)
                if len(path_parts) < 2: #if no file path there, and only top-level path
                    continue

                relative_path = path_parts[1]  #eg: src/main.py
                file_name = os.path.basename(relative_path) #main.py
                ext = os.path.splitext(file_name)[1].lower() #extract extension from filename
                path_segments = relative_path.split("/")


                #ignoring non code files and hidden build directories, we have created ignore list above
                if ext in IGNORED_EXTENSIONS:
                    continue

                #to ignore (.gitignore etc and files mentioned in IGNORED_EXACT_FILES)
                if file_name in IGNORED_EXACT_FILES or file_name.startswith("."):
                    continue

                # 3. Ignore forbidden directories anywhere in the file's path
                #path_segments[:-1] returns the last index of list of split list, last index contains the file name. "any()" returns true if any 1 is true 
                if any(part in IGNORED_DIRS or part.startswith(".") for part in path_segments[:-1]):
                    continue



                ##Reading file content into memory as a plain String
                with zf.open(zip_info) as file_data:
                                        
                    #open as string
                    file_content = file_data.read().decode("utf-8",errors="ignore") #ignore if not able to convert to utf-8, stops from crashing

                    if not file_content.strip():
                        continue #if file empty

                    ########################
                    #Splitting and Chunking#
                    ########################

                    #Language aware splitting 
                    if ext in EXTENSION_MAP:
                        splitter = RecursiveCharacterTextSplitter.from_language(
                            language=EXTENSION_MAP[ext],
                            chunk_size = 1000,
                            chunk_overlap = 150,
                        )

                    #else split with generic splitter
                    else:
                        splitter = RecursiveCharacterTextSplitter(
                            separators=[
                            "\nclass ", "\nstruct ", "\ninterface ", "\ntype ",
                            "\ndef ", "\nfunction ", "\nfunc ", "\nfn ", "\npub fn ",
                            "\n\n", "\n}", "\n", " ", ""
                        ],
                        chunk_size = 1000,
                        chunk_overlap = 150,
                        )

                    #create the document
                    raw_doc = Document(
                        page_content=file_content,
                        metadata={
                            "file_path": relative_path,
                            "file_name": file_name,
                            "file_type": ext.lstrip(".") #strips '.' from left side, if no '.' then does nothing
                            }
                        )

                    #split the docs
                    chunk = splitter.split_documents([raw_doc])
                    all_chunks.extend(chunk) #use extend instead of append to prevent forming: List[List[Document]]


        print(f"In memory processing done. Repository processed. Files converted into Chunks:{len(all_chunks)}")
        return all_chunks



    except Exception as e:
        print(f"Error in ingestion and splitting \n\n{e}")
        raise RuntimeError(f"Error in Ingestion and Splitting the public repository \n\n{e}")