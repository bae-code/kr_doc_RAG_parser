import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import faiss
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer


@dataclass
class ArticleChunk:
    article_id: str
    start_id: str
    end_id: str
    content: str
    paragraph_ids: List[str]
    relation_parts: List[str]

# HTML 불러오기
html_path = Path("test.html")
html_content = html_path.read_text(encoding="utf-8")
soup = BeautifulSoup(html_content, "html.parser")


def clean_text(text: str) -> str:
    """
    관련 조항을 찾기위한 텍스트 정제 함수
    - 띄워쓰기를 줄바꿈으로 변환
    - 특수기호 제거
    - 스페이스 압축
    """
    # ①~⑳ 까지 삭제
    text = text.replace("\n", " ")
    text = re.sub(r"[\u2460-\u2473]", "", text)  # ①②③ 특수기호 제거
    text = re.sub(r"\s+", " ", text)  # 스페이스 압축
    return text

def get_relation_parts(text: str) -> List[str]:
    """
    각 조항 내 관련된 조항을 찾아서 반환하는 함수

    Args:
        text (str): 파싱할 텍스트

    Returns:
        List[str]: 관련된 조항 목록
    """
    relation_part_pattern = re.compile(r"제\d+조(?:의\d+)?(?:제\d+항)?")
    matches = relation_part_pattern.findall(text)
    relation_parts = []
    for match in matches:
        relation_parts.append(match)
    return relation_parts

# 조문 단위로 파싱: 조문 시작 id부터 다음 조문 시작 전까지 포함
def parse_html_by_article_sections(soup: BeautifulSoup) -> List[ArticleChunk]:
    """
    조문 단위로 파싱: 조문 시작 id부터 다음 조문 시작 전까지 포함

    Args:
        soup (BeautifulSoup): 파싱할 HTML 파일

    Returns:
        List[Dict]: 조문 단위로 파싱된 데이터
        
    """
    all_p_tags = soup.find_all("p")
    article_indices = []

    # 조문 시작점을 가진 <p> 태그 인덱스 및 article_id 저장
    for i, tag in enumerate(all_p_tags):
        text = tag.get_text(strip=True)
        if re.match(r"^제\d+조(?:의\d+)?", text):
            article_id = re.match(r"^(제\d+조(?:의\d+)?)", text).group(1)
            article_indices.append((i, article_id))

    chunks = []

    for idx, (start_idx, article_id) in enumerate(article_indices):
        end_idx = article_indices[idx + 1][0] if idx + 1 < len(article_indices) else len(all_p_tags)
        content_parts = []
        id_list = []
        
        relation_parts = []
        for tag in all_p_tags[start_idx:end_idx]:
            content = tag.get_text(separator=" ", strip=True).replace("<br />", "")
            content_parts.append(content)
            clean_content = clean_text(content)
            id_list.append(tag.get("id", "unknown"))
            relation_parts.extend(get_relation_parts(clean_content))


        chunks.append(ArticleChunk(
            article_id=article_id,
            start_id=all_p_tags[start_idx].get("id", "unknown"),
            end_id=all_p_tags[end_idx - 1].get("id", "unknown"),
            content="\n".join(content_parts),
            paragraph_ids=id_list,
            relation_parts=relation_parts[1:] # 첫번째 요소는 관련조문이 아니기 때문에 제거
        ))

    return chunks

# 실행
article_chunks = parse_html_by_article_sections(soup)


model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

content_list = []   
for chunk in article_chunks:
    content_list.append(chunk.content)
    print(f"조문 ID: {chunk.article_id}")
    print(f"시작 ID: {chunk.start_id} / 종료 ID: {chunk.end_id}")
    print(f"포함된 문단 ID: {chunk.paragraph_ids}")
    print("-" * 60)


vector = model.encode(content_list, normalize_embeddings=True)

dim = vector.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(vector)

#### Search Test ####

query = ["허위 신고 과태료"]

query_vector = model.encode(query, normalize_embeddings=True)

top_k = 3
D, I = index.search(query_vector, k=top_k)

for i in range(top_k):
    chunk = article_chunks[I[0][i]]
    print(f"검색 결과 {i+1}: {chunk.content}")
    print(chunk.relation_parts)
    print(f"점수: {D[0][i]}")
    print("-" * 60)
