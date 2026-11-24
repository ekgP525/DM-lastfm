from lastfm_rec.tables import check_consistency, compare_table, to_markdown


def _result(ndcg, seeds=(0, 1), n_users=100, k=10):
    summ = {key: {"mean": 0.1, "std": 0.01} for key in
            ["recall@10", "coverage", "tail_ratio", "avg_popularity", "n_eval_users"]}
    summ["ndcg@10"] = {"mean": ndcg, "std": 0.01}
    runs = [{"seed": s, "fit_sec": 2.0, "rec_sec_per_user": 0.001,
             "all": {"n_eval_users": n_users}, "reachable": {"n_eval_users": n_users}} for s in seeds]
    return {"k": k, "stage": "val", "params": {}, "runs": runs, "summary": {"all": summ, "reachable": summ}}


def test_table_rows_and_timing():
    df = compare_table({"A": _result(0.1), "B": _result(0.2)}, "all")
    assert list(df.index) == ["A", "B"]
    assert df.loc["B", "NDCG@10 mean"] == 0.2
    assert df.loc["A", "Fit sec"] == 2.0 and df.loc["A", "Rec ms/user"] == 1.0


def test_markdown_bolds_best_accuracy_only():
    df = compare_table({"A": _result(0.1), "B": _result(0.2)}, "all")
    lines = to_markdown(df, k=10).splitlines()
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = {r.split("|")[1].strip(): [c.strip() for c in r.strip("|").split("|")] for r in lines[2:]}
    ndcg, cov = header.index("NDCG@10"), header.index("Coverage")
    assert rows["B"][ndcg] == "**0.2000 ± 0.0100**"
    assert rows["A"][ndcg] == "0.1000 ± 0.0100"
    # 다양성 지표(Coverage 등)는 값이 같아도 굵게 표시하지 않는다
    assert "**" not in rows["A"][cov] and "**" not in rows["B"][cov]


def test_consistency_flags_mismatched_runs():
    ok = {"A": _result(0.1), "B": _result(0.2)}
    assert check_consistency(ok) == []
    bad = {"A": _result(0.1), "B": _result(0.2, seeds=(0,), n_users=90)}
    issues = check_consistency(bad)
    assert any("seeds" in i for i in issues) and any("n_eval_users" in i for i in issues)
