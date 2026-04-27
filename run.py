from datasets import load_dataset
from collections import defaultdict
from repo_kg_builder import RepoKGBuilder
from kg_query import KGQueryEngine


def get_latest_commit(instances, repo):
    commits = [i['base_commit'] for i in instances if i['repo'] == repo]
    return commits[-1] if commits else None


def main():
    print("Loading SWE-bench Lite...")
    ds = load_dataset('princeton-nlp/SWE-bench_Lite', split='test')
    instances = list(ds)

    # Show available repos
    repo_counts = defaultdict(int)
    for i in instances:
        repo_counts[i['repo']] += 1
    print("\nAvailable repos:")
    for repo, count in sorted(repo_counts.items()):
        print(f"  {repo} ({count} issues)")

    repo = input("\nEnter repo to build KG for (e.g. psf/requests): ").strip()
    commit = get_latest_commit(instances, repo)
    if not commit:
        print(f"Repo '{repo}' not found in dataset.")
        return

    print(f"\nUsing commit: {commit}")
    builder = RepoKGBuilder()
    kg = builder.build(repo, commit)
    builder.save(repo, kg)

    # Quick summary
    engine = KGQueryEngine(kg)
    from collections import Counter
    types = Counter(n['type'] for n in kg['nodes'])
    relations = Counter(e['relation'] for e in kg['edges'])
    print("\nNode types:", dict(types))
    print("Edge relations:", dict(relations))


if __name__ == '__main__':
    main()
