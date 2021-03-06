import streamlit as st
import pybaseball as pb
import pandas as pd
import scipy.stats as stat
import random
import pickle
import numpy as np
import plotly.express as px
import os
import itertools
import plotly.graph_objects as go

# 定数
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
YEARS = [2017, 2018, 2019, 2020]
with open(f"{APP_ROOT}/resources/svm.pickle", "rb") as f:
    SVM = pickle.load(f)
with open(f"{APP_ROOT}/resources/expect.pickle", "rb") as f:
    BB_TYPE_COUNT = pickle.load(f)
LAUNCH_SPEED_RANGE = np.linspace(0, 150, 150)
LAUNCH_ANGLE_RANGE = np.linspace(-90, 90, 180)
LABEL_MAP = {
    "single": "単打", "double": "二塁打", "triple": "三塁打", "home_run": "本塁打", "field_out": "凡退",
    "avg": "打率", "obp": "出塁率", "slg": "長打率", "ops": "OPS (出塁率＋長打率)"
}

# メッセージ
INFO = """
- `Statcast` データを用いたMLBバッティングシミュレーターです。
- シミュレーションは下記手順で実行出来ます。
    - サイドバーでプレーヤー名と、シミュレーションを行うデータ取得年を設定して下さい。
    - `シミュレーション` ボタンを押下して下さい。データが取得され、シミュレーションが行われます。
    - インプレー回数は一律500回でシミュレーションを行っています。
"""
WARN = """
### :warning: **注意** :warning:

- 打席数200を下回るとシミュレーションを行いません。
- シミュレーションを行うデータの取得には時間がかかります。
    - 対策として、アプリでは同一データでの再検索時にはキャッシュを利用しています。
- 条件でデータを得られなかった場合はエラーメッセージを表示します。条件を修正して再検索を行って下さい。
"""
PLAYERID_ERROR = """
指定の選手が存在しませんでした。
姓・名のスペルが合っているか、姓・名を逆に入力していないかを確認して下さい。
"""
FROM_PLAYERID_ERROR = """
指定の比較元の選手が存在しませんでした。
姓・名のスペルが合っているか、姓・名を逆に入力していないかを確認して下さい。
"""
TO_PLAYERID_ERROR = """
指定の比較元の選手が存在しませんでした。
姓・名のスペルが合っているか、姓・名を逆に入力していないかを確認して下さい。
"""
STATCAST_ERROR = """
条件に合う `Statcast` データが存在しませんでした。
対象選手が対象シーズンにプレーしているか確認して下さい。
"""
FROM_STATCAST_ERROR = """
条件に合う比較元 `Statcast` データが存在しませんでした。
対象選手が対象シーズンにプレーしているか確認して下さい。
"""
TO_STATCAST_ERROR = """
条件に合う比較元 `Statcast` データが存在しませんでした。
対象選手が対象シーズンにプレーしているか確認して下さい。
"""

@st.cache(suppress_st_warning=True)
def __search_playerid(first_name, last_name):
    players = pd.read_csv(f"{APP_ROOT}/resources/players.csv")
    info = players[
        (players["name_first"].str.upper() == first_name.upper()) & (players["name_last"].str.upper() == last_name.upper())
    ].sort_values(["mlb_played_last"], ascending=False)
    return info["key_mlbam"].values

@st.cache(suppress_st_warning=True)
def __get_statcast_data(start_dt, end_dt, player_id):
    return pb.statcast_batter(start_dt, end_dt, player_id)

@st.cache(suppress_st_warning=True)
def __get_bb_k_rate(first_name, last_name, year):
    # K・BB%は不変を仮定
    bs = __get_batting_stats(year)
    bb_k_rate = bs[bs["Name"] == f"{first_name} {last_name}"]
    bb_rate = bb_k_rate["BB%"].values[0]
    k_rate = bb_k_rate["K%"].values[0]
    return bb_rate, k_rate

@st.cache(suppress_st_warning=True)
def __get_batting_stats(year):
    return pb.batting_stats(f"{year}", qual=200, stat_columns=["NAME", "BB_PCT", "K_PCT"])

def __simulate(df, first_name, last_name, year):
    df = df[(df["launch_speed"].isnull() == False) & (df["launch_angle"].isnull() == False) & (df["launch_speed_angle"].isnull() == False)]
    df = df[
        df["events"].isin(["home_run", "field_out", "grounded_into_double_play", "single", "double_play", "double", "triple", "triple_play"])
    ]
    df["events"] = df["events"].replace({
        "grounded_into_double_play": "field_out",
        "double_play": "field_out",
        "triple_play": "field_out"
    })

    ls = df["launch_speed"].values
    la = df["launch_angle"].values
    lsa, lsloc, lsscale = stat.skewnorm.fit(ls)
    laa, laloc, lascale = stat.skewnorm.fit(la)

    sim = pd.DataFrame(columns=["pattern", "ls", "la"])
    for i in range(0, 100):
        pred_ls = stat.skewnorm.rvs(lsa, lsloc, lsscale, size=500)
        pred_la = stat.skewnorm.rvs(laa, laloc, lascale, size=500)

        pred_ls = random.sample(list(pred_ls), len(list(pred_ls)))
        pred_la = random.sample(list(pred_la), len(list(pred_la)))
        d = pd.DataFrame(columns=["pattern", "ls", "la"])
        d["ls"] = pred_ls
        d["la"] = pred_la
        d["pattern"] = i
        sim = pd.concat([sim, d])
    
    sim_lsa = SVM.predict(sim[["ls", "la"]])
    sim["launch_speed_angle"] = sim_lsa

    sim_by_p_lsa = sim.groupby(["pattern", "launch_speed_angle"]).count().reset_index()[["pattern", "launch_speed_angle", "ls"]].rename(columns={"ls": "count"})
    sim_bb_by_p_lsa = pd.merge(sim_by_p_lsa,BB_TYPE_COUNT, on="launch_speed_angle").rename(columns={"count_x": "count"})[[
        "pattern", "launch_speed_angle", "count", "events", "percentage"
    ]]
    sim_bb_by_p_lsa["predict"] = sim_bb_by_p_lsa["count"] * sim_bb_by_p_lsa["percentage"]
    sim_vertical = sim_bb_by_p_lsa.groupby(["pattern", "events"]).sum().reset_index()[["pattern", "events", "predict"]]
    p = sim_vertical.pivot_table(values=["predict"], index="pattern", columns=["events"]).reset_index()["predict"].reset_index()

    bb_rate, k_rate = __get_bb_k_rate(first_name, last_name, year)

    p["pa"] = 500 / (1 - bb_rate - k_rate)
    p["bb"] = p["pa"] * bb_rate
    p["so"] = p["pa"] * k_rate
    p["ab"] = 500 + p["so"]
    p["hits"] = p["single"] + p["double"] + p["triple"] + p["home_run"]
    p["tb"] = p["single"] + p["double"] * 2 + p["triple"] * 3 + p["home_run"] * 4

    p["avg"] = p["hits"] / p["ab"]
    p["obp"] = (p["hits"] + p["bb"]) / p["pa"]
    p["slg"] = p["tb"] / p["ab"]
    p["ops"] = p["obp"] + p["slg"]

    describe = p.describe()[[
        "single", "double", "triple", "home_run", "avg", "obp", "slg", "ops"
    ]].rename(columns=LABEL_MAP)

    return lsa, lsloc, lsscale, laa, laloc, lascale, sim, sim_vertical, describe


def simulate(first_name, last_name, year):
    player_ids = __search_playerid(first_name, last_name)
    if (len(player_ids) == 0):
        st.error(PLAYERID_ERROR)
        return
    
    player_id = str(player_ids[0]).split(".")[0]
    df = __get_statcast_data(f"{year}-01-01", f"{year}-12-31", player_id)

    if (len(df) == 0):
        st.error(STATCAST_ERROR)
        return

    lsa, lsloc, lsscale, laa, laloc, lascale, sim, sim_vertical, describe = __simulate(df, first_name, last_name, year)

    st.markdown(f"# {first_name} {last_name}, {year}")

    with st.beta_container():
        st.markdown("## シミュレーション結果")
        st.table(describe.query("index in ['mean', '50%', 'min', 'max']").rename(
            index={"mean": "平均値", "min": "最小値", "max": "最大値", "50%": "中央値"}
        ))

    with st.beta_container():
        st.markdown("## シミュレーショングラフ")
        fig = px.line(
            sim_vertical.replace(LABEL_MAP), x="pattern", y="predict", color="events",
            labels={
                "pattern": "試行回", "predict": "シミュレーション値（単位: 本）"
            }
        )
        fig.layout["legend"]["title"]["text"] = "結果"
        st.plotly_chart(fig, use_container_width=True)
    
    with st.beta_container():
        st.markdown("## シミュレーション打球プロット")
        fig = px.scatter(
            sim, x="ls", y="la", color="launch_speed_angle",
            labels={
                "ls": "打球速度", "la": "打球角度"
            }
        )
        fig.layout["legend"]["title"]["text"] = "打球種別"
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 累積分布関数")
    ls_column, la_column = st.beta_columns(2)
    with ls_column:
        st.markdown("### 打球速度")
        lsy = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_SPEED_RANGE, lsa, lsloc, lsscale))]
        fig = px.line(
            x=LAUNCH_SPEED_RANGE, y=lsy, labels={
                "x": "打球速度", "y": "確率"
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with la_column:
        st.markdown("### 打球角度")
        lay = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_ANGLE_RANGE, laa, laloc, lascale))]
        fig = px.line(
            x=LAUNCH_ANGLE_RANGE, y=lay, labels={
                "x": "打球角度", "y": "確率"
            }
        )
        st.plotly_chart(fig, use_container_width=True)


def compare(from_first_name, from_last_name, from_year, to_first_name, to_last_name, to_year):
    from_player_ids = __search_playerid(from_first_name, from_last_name)
    if (len(from_player_ids) == 0):
        st.error(FROM_PLAYERID_ERROR)
        return
    to_player_ids = __search_playerid(to_first_name, to_last_name)
    if (len(to_player_ids) == 0):
        st.error(TO_PLAYERID_ERROR)
        return
    
    from_player_id = str(from_player_ids[0]).split(".")[0]
    to_player_id = str(to_player_ids[0]).split(".")[0]
    
    from_df = __get_statcast_data(f"{from_year}-01-01", f"{from_year}-12-31", from_player_id)
    if (len(from_df) == 0):
        st.error(FROM_STATCAST_ERROR)
        return
    
    to_player_id = str(to_player_ids[0]).split(".")[0]
    to_df = __get_statcast_data(f"{to_year}-01-01", f"{to_year}-12-31", to_player_id)
    if (len(to_df) == 0):
        st.error(TO_STATCAST_ERROR)
        return
    
    from_lsa, from_lsloc, from_lsscale, from_laa, from_laloc, from_lascale, from_sim, from_sim_vertical, from_describe = __simulate(from_df, from_first_name, from_last_name, from_year)
    to_lsa, to_lsloc, to_lsscale, to_laa, to_laloc, to_lascale, to_sim, to_sim_vertical, to_describe = __simulate(to_df, to_first_name, to_last_name, to_year)

    st.markdown("## シミュレーション結果")
    sim_left, sim_right = st.beta_columns(2)
    with sim_left:
        st.markdown(f"### {from_first_name} {from_last_name}, {from_year}")
        st.table(from_describe.query("index in ['mean', '50%', 'min', 'max']").rename(
            index={"mean": "平均値", "min": "最小値", "max": "最大値", "50%": "中央値"}
        ))

    with sim_right:
        st.markdown(f"### {to_first_name} {to_last_name}, {to_year}")
        st.table(to_describe.query("index in ['mean', '50%', 'min', 'max']").rename(
            index={"mean": "平均値", "min": "最小値", "max": "最大値", "50%": "中央値"}
        ))

    st.markdown("## シミュレーション比較グラフ")
    with st.beta_container():
        # describeのマージ
        from_mean = from_describe.query("index == 'mean'").reset_index()
        to_mean = to_describe.query("index == 'mean'").reset_index()
        from_mean["name"] = f"{from_first_name} {from_last_name}, {from_year}"
        to_mean["name"] = f"{to_first_name} {to_last_name}, {to_year}"
        from_mean = from_mean.set_index("name")
        to_mean = to_mean.set_index("name")
        m = pd.concat([from_mean, to_mean])
        target = list(m.columns)
        target.remove("index")
        m = m[target]
        mn=(m / m.sum()) * 100
        mnt = mn.T.reset_index()
        mntm = pd.melt(mnt, id_vars="events", var_name="name", value_name="val")
        fig = px.bar(
            mntm, x="val", y="events", color="name",
            labels={
                "val": "比率", "events": "イベント"
            }
        )
        fig = fig.update_traces(hovertemplate=None)
        fig = fig.update_layout(hovermode="x", legend_title="選手名")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## シミュレーショングラフ")
    sim_graph_left, sim_graph_right = st.beta_columns(2)
    with sim_graph_left:
        st.markdown(f"### {from_first_name} {from_last_name}, {from_year}")
        fig = px.line(
            from_sim_vertical.replace(LABEL_MAP), x="pattern", y="predict", color="events", range_y=[0, 500],
            labels={
                "pattern": "試行回", "predict": "シミュレーション値（単位: 本）"
            }
        )
        fig.layout["legend"]["title"]["text"] = "結果"
        st.plotly_chart(fig, use_container_width=True)

    with sim_graph_right:
        st.markdown(f"### {to_first_name} {to_last_name}, {to_year}")
        fig = px.line(
            to_sim_vertical.replace(LABEL_MAP), x="pattern", y="predict", color="events", range_y=[0, 500],
            labels={
                "pattern": "試行回", "predict": "シミュレーション値（単位: 本）"
            }
        )
        fig.layout["legend"]["title"]["text"] = "結果"
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("## シミュレーション打球プロット")
    sim_bat_graph_left, sim_bat_graph_right = st.beta_columns(2)
    with sim_bat_graph_left:
        st.markdown(f"### {from_first_name} {from_last_name}, {from_year}")
        fig = px.scatter(
            from_sim, x="ls", y="la", color="launch_speed_angle",
            labels={
                "ls": "打球速度", "la": "打球角度"
            }
        )
        fig.layout["legend"]["title"]["text"] = "打球種別"
        st.plotly_chart(fig, use_container_width=True)
    
    with sim_bat_graph_right:
        st.markdown(f"### {to_first_name} {to_last_name}, {to_year}")
        fig = px.scatter(
            to_sim, x="ls", y="la", color="launch_speed_angle",
            labels={
                "ls": "打球速度", "la": "打球角度"
            }
        )
        fig.layout["legend"]["title"]["text"] = "打球種別"
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("## 累積分布関数")
    ls_column, la_column = st.beta_columns(2)
    with ls_column:
        st.markdown("### 打球速度")
        from_lsy = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_SPEED_RANGE, from_lsa, from_lsloc, from_lsscale))]
        to_lsy = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_SPEED_RANGE, to_lsa, to_lsloc, to_lsscale))]
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=LAUNCH_SPEED_RANGE,
            y=from_lsy,
            mode="lines",
            name=f"{from_first_name} {from_last_name}, {from_year}"
        ))
        fig.add_trace(go.Scatter(
            x=LAUNCH_SPEED_RANGE,
            y=to_lsy,
            mode="lines",
            name=f"{to_first_name} {to_last_name}, {to_year}"
        ))
        fig.update_layout(
            xaxis_title="打球速度",
            yaxis_title="確率",
            legend_title="選手名, 年度"
        )
        st.plotly_chart(fig, use_container_width=True)

    with la_column:
        st.markdown("### 打球角度")
        from_lay = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_ANGLE_RANGE, from_laa, from_laloc, from_lascale))]
        to_lay = [i for i in itertools.accumulate(stat.skewnorm.pdf(LAUNCH_ANGLE_RANGE, to_laa, to_laloc, to_lascale))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=LAUNCH_ANGLE_RANGE,
            y=from_lay,
            mode="lines",
            name=f"{from_first_name} {from_last_name}, {from_year}"
        ))
        fig.add_trace(go.Scatter(
            x=LAUNCH_ANGLE_RANGE,
            y=to_lay,
            mode="lines",
            name=f"{to_first_name} {to_last_name}, {to_year}"
        ))
        fig.update_layout(
            xaxis_title="打球角度",
            yaxis_title="確率",
            legend_title="選手名, 年度"
        )
        st.plotly_chart(fig, use_container_width=True)

def set_single_search_sidebar():
    st.sidebar.markdown("[compareモードへ](https://share.streamlit.io/prs-watch/simcast/main/app.py?mode=compare)")
    st.sidebar.markdown("# :mag_right: シミュレーション条件")

    st.sidebar.markdown("## :baseball: 選手情報")
    first_name = st.sidebar.text_input("名", "Trea")
    last_name = st.sidebar.text_input("姓", "Turner")

    st.sidebar.markdown("## :calendar: 対象シーズン")
    year = st.sidebar.selectbox("シーズン", YEARS)

    if st.sidebar.button("シミュレーション"):
        with st.spinner("シミュレーション中.."):
            simulate(first_name, last_name, year)
    
def set_compare_sidebar():
    st.sidebar.markdown("[シミュレーションモードへ](https://share.streamlit.io/prs-watch/simcast/main/app.py)")
    st.sidebar.markdown("# :mag_right: シミュレーション条件")

    st.sidebar.markdown("## 比較元")
    from_first_name = st.sidebar.text_input("名", "Trea")
    from_last_name = st.sidebar.text_input("姓", "Turner")
    from_year = st.sidebar.selectbox("シーズン", YEARS, index=0)

    st.sidebar.markdown("## 比較先")
    to_first_name = st.sidebar.text_input("名", "Aaron")
    to_last_name = st.sidebar.text_input("姓", "Judge")
    to_year = st.sidebar.selectbox("シーズン", YEARS, index=1)

    if st.sidebar.button("シミュレーション"):
        with st.spinner("シミュレーション中.."):
            compare(from_first_name, from_last_name, from_year, to_first_name, to_last_name, to_year)


# メタ情報
st.set_page_config(
    page_title="Simcast",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon=f"{APP_ROOT}/resources/icon.jpg"
)

# スタイル
st.markdown("""
    <style>
    .css-1y0tads {
        padding: 0rem 5rem 10rem
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# タイトル
st.title("Simcast")
st.subheader("Created by [@hctaw_srp](https://twitter.com/hctaw_srp)")

# 固定文言
st.markdown(INFO)
st.warning(WARN)

params = st.experimental_get_query_params()
if not params:
    set_single_search_sidebar()
else:
    if len(params["mode"]) != 0 and params["mode"][0] == "compare":
        set_compare_sidebar()
    else:
        set_single_search_sidebar()