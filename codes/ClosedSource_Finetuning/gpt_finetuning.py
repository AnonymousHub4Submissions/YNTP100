import json
import os
import time
from pathlib import Path
from openai import OpenAI

api_key = os.environ["OPENAI_API_KEY_FT"]
client = OpenAI(api_key=api_key)

def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def wait_file_processed(file_id: str, sleep_sec: int = 5, max_wait_sec: int = 600):
    start = time.time()
    while True:
        f = client.files.retrieve(file_id)
        status = getattr(f, "status", None)
        if status == "processed":
            return
        if time.time() - start > max_wait_sec:
            raise TimeoutError(f"File not processed in time: {file_id} (status={status})")
        print(f"[file] waiting... file_id={file_id} status={status}")
        time.sleep(sleep_sec)

def upload_file(user_id: str, jsonl_path: Path, base_model: str, log_path: Path):
    uploaded = client.files.create(
        file=jsonl_path.open("rb"),
        purpose="fine-tune",
    )
    file_id = uploaded.id
    print(f"[upload] user={user_id} file_id={file_id} path={jsonl_path}")

    append_jsonl(log_path, {
        "event": "uploaded",
        "user_id": user_id,
        "file_id": file_id,
        "path": str(jsonl_path),
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    wait_file_processed(file_id)

    return file_id

def submit_one(user_id: str, file_id: str, base_model: str, log_path: Path):
    job = client.fine_tuning.jobs.create(
        training_file=file_id,
        model=base_model,
        suffix=user_id,
    )
    job_id = job.id
    print(f"[job] submitted user={user_id} job_id={job_id} base_model={base_model}")

    append_jsonl(log_path, {
        "event": "job_submitted",
        "user_id": user_id,
        "job_id": job_id,
        "file_id": file_id,
        "base_model": base_model,
        "suffix": user_id,
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    return job_id

def poll_until_done(job_id: str, user_id: str, log_path: Path, sleep_sec: int = 30):
    while True:
        job = client.fine_tuning.jobs.retrieve(job_id)
        status = job.status
        fine_tuned_model = getattr(job, "fine_tuned_model", None)
        print(f"[job] user={user_id} job_id={job_id} status={status} model={fine_tuned_model}")

        if status in ("succeeded", "failed", "cancelled"):
            append_jsonl(log_path, {
                "event": "job_finished",
                "user_id": user_id,
                "job_id": job_id,
                "status": status,
                "fine_tuned_model": fine_tuned_model,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            })
            return status, fine_tuned_model

        time.sleep(sleep_sec)

def load_ft_state(log_path: str):
    ft_state = {}

    if not log_path.exists():
        return ft_state

    with log_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(f"[warn] bad json at line {line_no}: {line[:120]}")
                continue

            user_id = event.get("user_id")
            if not user_id:
                continue

            st = ft_state.setdefault(user_id, {
                "status": None,
                "file_id": None,
                "job_id": None
            })

            e = event.get("event")
            if e == "uploaded":
                st["status"] = "uploaded"
                st["file_id"] = event.get("file_id") or st.get("file_id")
            elif e == "job_submitted":
                st["status"] = "submitted"
                st["job_id"] = event.get("job_id") or st.get("job_id")
                st["file_id"] = event.get("file_id") or st.get("file_id")
            elif e == "job_finished":
                st["status"] = event.get("status") or st.get("status")
                st["job_id"] = event.get("job_id") or st.get("job_id")
                st["model_name"] = event.get("fine_tuned_model") or st.get("model_name")

    return ft_state

if __name__ == "__main__":
    # gpt-3.5-turbo, gpt-4o-mini-2024-07-18
    base_model = "gpt-4o-mini-2024-07-18"
    log_path = Path(f"./log_{base_model}.jsonl")
    ft_state = load_ft_state(log_path)

    langs = ["jp", "en", "cn"]
    data_dir = Path("./dataset")
    jsonl_files = []
    for lang in langs:
        jsonl_files.extend(sorted((data_dir/lang).glob("*.jsonl")))

    model_name_path = Path(f"./model_name_{base_model}.json")
    model_name_dict = {}
    for jsonl_file in jsonl_files:
        user_id = jsonl_file.stem
        state = ft_state.get(user_id)

        if state is None:
            file_id = upload_file(user_id, jsonl_file, base_model, log_path)
            job_id = submit_one(user_id, file_id, base_model, log_path)
        elif state["status"] == "succeeded":
            print(f"[skip] user_id={user_id}, model_name={state["model_name"]}")
            model_name_dict[user_id] = state["model_name"]
            continue
        elif state["status"] in ["uploaded", "failed", "cancelled"]:
            file_id = state["file_id"]
            job_id = submit_one(user_id, file_id, base_model, log_path)
        elif state["status"] == "submitted":
            job_id = state["job_id"]

        status, model_name = poll_until_done(job_id, user_id, log_path)
        if status == "succeeded":
            model_name_dict[user_id] = model_name

    with model_name_path.open("w", encoding="utf-8") as f:
        json.dump(model_name_dict, f, ensure_ascii=False, indent=4)
