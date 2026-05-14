from huggingface_hub import HfApi

api = HfApi()

api.create_repo(repo_id="Prifea/hit-predictor", repo_type="model")

api.upload_file(
    path_or_fileobj=r"C:\Users\DELL\Desktop\what-makes-a-hit\app\what-makes-a-hit-model.pkl",
    path_in_repo="what-makes-a-hit-model.pkl",
    repo_id="Prifea/hit-predictor",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=r"C:\Users\DELL\Desktop\what-makes-a-hit\app\le_genre.pkl",
    path_in_repo="le_genre.pkl",
    repo_id="Prifea/hit-predictor",
    repo_type="model"
)

api.upload_file(
    path_or_fileobj=r"C:\Users\DELL\Desktop\what-makes-a-hit\app\le_explicit_genre.pkl",
    path_in_repo="le_explicit_genre.pkl",
    repo_id="Prifea/hit-predictor",
    repo_type="model"
)

print("All files uploaded!")