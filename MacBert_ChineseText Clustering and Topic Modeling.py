from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')
from bertopic import BERTopic
from umap import UMAP
import pandas as pd
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from hdbscan import HDBSCAN

## 5.1 加载数据
filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\text\\fudan_text_cn.csv'
dataset = pd.read_csv(filepath,  encoding='gbk')

# 查看 metadata 数据
abstracts = dataset["Abstracts"]
titles = dataset["Titles"]

# abstracts[:1],
titles[:1]
abstracts[:1]

# 加载 MacBERT
model_name = "hfl/chinese-macbert-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 生成嵌入
def get_macbert_embeddings(texts, batch_size=32):
    model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)

embeddings = get_macbert_embeddings(abstracts.tolist(), batch_size=32)

#接下来，我们将嵌入降维到2维，以便我们可以绘制它们并对生成的聚类有一个大致的了解。
# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization
umap_model = UMAP(n_components=5, min_dist=0.0, metric='cosine', random_state=42)
reduced_embeddings = umap_model.fit_transform(embeddings)

### 5.2.3 根据降维后的 embedding 进行聚类
#第一种HDBSCAN
hdbscan_model = HDBSCAN(
    min_cluster_size=2,  # Reduce cluster size requirement
    min_samples=2,  # Lower noise threshold
    metric='euclidean',  # Better for high-dimensional embeddings
    cluster_selection_method='leaf',  # More fine-grained clustering
    prediction_data=True
).fit(reduced_embeddings)
clusters = hdbscan_model.labels_ #所有文档的分类情况
len(set(clusters))

#第二种 KMeans
# kmeans = KMeans(n_clusters=5, random_state=42).fit(reduced_embeddings)
# clusters = kmeans.labels_
# len(set(clusters))


### 5.2.4 检查聚类结果
# 打印 cluster 0 中的前三个文档
cluster = 1
for index in np.where(clusters==cluster)[0][:3]:  #[0]返回一个True的数组的具体位置（刚好等于第几个文档）
    print(abstracts[index][:300] + "... \n")

#接下来，我们将嵌入降维到2维，以便我们可以绘制它们并对生成的聚类有一个大致的了解。
# Reduce 384-dimensional embeddings to 2 dimensions for easier visualization
reduced_embeddings = UMAP(
    n_components=2, min_dist=0.0, metric='cosine', random_state=42
).fit_transform(embeddings)

# Create dataframe
df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
df["title"] = titles
df["cluster"] = [str(c) for c in clusters]

# Select outliers and non-outliers (clusters)
clusters_df = df.loc[df.cluster != "-1", :]
outliers_df = df.loc[df.cluster == "-1", :]

### 5.2.5 静态绘图
# 分别绘制离群点和非离群点
# 解释：alpha 是透明度，s 是点的大小，cmap 是颜色图
plt.scatter(outliers_df.x, outliers_df.y, alpha=0.05, s=2, c="grey")
plt.scatter(
    clusters_df.x, clusters_df.y, c=clusters_df.cluster.astype(int),
    alpha=0.6, s=2, cmap='tab20b'
)
plt.show()
# plt.savefig("matplotlib.png", dpi=300)  # Uncomment to save the graph as a .png



### 5 BERTopic: 一个模块化的主题建模框架
topic_model = BERTopic(
    embedding_model=model,
    umap_model=umap_model,
    hdbscan_model= hdbscan_model ,  #hdbscan_model 或者 kmeans
    verbose=False
).fit(abstracts, embeddings)

topic_model_info = topic_model.get_topic_info()
topic_model_info

#> TF-IDF 是词频-逆文档频率（Term Frequency-Inverse Document Frequency）的缩写，是一种用于评估词语在文档集合中的重要性的统计方法。
#> c-TF-IDF 是类-词频-逆文档频率（Class-based TF-IDF）的缩写，是一种在主题建模中常用的加权方法，它考虑了文档类别对词语重要性的影响。
topic_model.get_topic(1)
topic_model_abstracts =  topic_model.get_document_info(abstracts)[:5]
topic_model_abstracts

### 5.3.2 可视化
import plotly.io as pio
# Visualize topics and documents
pio.renderers.default = "browser"
fig = topic_model.visualize_documents(
    titles,
    reduced_embeddings=reduced_embeddings,
    width=1200,
    hide_annotations=True
)
fig.show()
# Update fonts of legend for easier visualization
fig.update_layout(font=dict(size=16))
# 可视化带有排名关键词的条形图
topic_model.visualize_barchart().show()
# 可视化主题之间的关系
topic_model.visualize_heatmap(n_clusters=1).show() #lower HDBSCAN小于n_clusters-1, Kmeans 小于n_clusters
# 可视化主题的潜在层次结构
topic_model.visualize_hierarchy().show()
#越上层，主题越抽象，越下层，主题越具体，两个进行连线的时候表示两者是相似的


# Update topics with more words for better visualization
topic_model.update_topics(abstracts, top_n_words=500)
from wordcloud import WordCloud
import matplotlib.pyplot as plt
def create_wordcloud(model, topic, max_words=1000, width=1600, height=800):
    """Generate and display a word cloud for a given topic."""
    plt.figure(figsize=(12, 6))

    # Extract top words and their importance
    topic_words = model.get_topic(topic)

    # Convert topic words to dictionary format
    if isinstance(topic_words, list) and all(isinstance(i, tuple) for i in topic_words):
        word_freq = dict(topic_words)
    else:
        raise ValueError("Unexpected format from topic_model.get_topic(). Expected list of (word, value) tuples.")

    # Generate word cloud
    wordcloud = WordCloud(
        background_color="white",
        max_words=max_words,
        width=width,
        height=height,
        colormap="tab10",
        font_path='simhei.ttf'
    ).generate_from_frequencies(word_freq)

    # Display the word cloud
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

# Test the function
create_wordcloud(topic_model, topic=1)


