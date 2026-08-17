# site — 결과 대시보드

`https://minjaejun.github.io/semicon-ai-lab/`

의존성 없는 정적 페이지다. 번들러도, CDN 도, 프레임워크도 쓰지 않는다.
차트는 `chart.js` 안의 SVG 생성 함수 몇 개가 전부다.

## 갱신 방법

```bash
python build_all.py            # 데이터 생성 + 분석
python site/build_site_data.py # 산출물 -> site/data.json
python site/_verify.py         # data.json 과 app.js 참조 정합성 검사
```

`build_site_data.py` 는 30만 행을 그대로 올리지 않는다.
산점도는 층화 샘플 2,000점 이하, 시계열은 일 단위 집계본, 표는 상위 N개만 남겨
`data.json` 하나로 압축한다. 결과는 약 150KB.

## 파일

| 파일 | 역할 |
|---|---|
| `index.html` | 셸 + 네비게이션 |
| `app.js` | 페이지 6개 렌더링 (ES module) |
| `chart.js` | SVG 차트: 산점도 · 시계열 · 막대 · 웨이퍼맵 |
| `style.css` | 다크 테마 |
| `data.json` | 대시보드용 데이터 (생성물) |
| `build_site_data.py` | 산출물 → data.json |
| `_verify.py` | 정적 검증 |
