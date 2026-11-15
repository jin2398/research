"""Regression tests for shared HTML storefront + data substitution."""
import json
import tempfile
import unittest
from pathlib import Path

from src.dashboard.storefront import storefront_html

ROOT=Path(__file__).resolve().parents[1]


class StorefrontTests(unittest.TestCase):
    def test_demo_standalone_is_the_same_html(self):
        html,source=storefront_html(ROOT,embedded=False)
        self.assertEqual(source,'demo')
        self.assertEqual(html,(ROOT/'codisay_preview.html').read_text(encoding='utf-8'))

    def test_embed_keeps_html_and_adds_management_navigation(self):
        html,source=storefront_html(ROOT,embedded=True)
        self.assertEqual(source,'demo')
        self.assertIn("link.href='?page='+pair[1]",html)
        self.assertIn("link.target='_blank'",html)
        self.assertIn('원하는 스타일,',html)
        self.assertIn('제품',html.replace('추천 상품 둘러보기','제품'))

    def test_csv_and_real_metrics_are_used_without_fake_ratings(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'data').mkdir()
            (root/'docs').mkdir()
            (root/'codisay_preview.html').write_text((ROOT/'codisay_preview.html').read_text(encoding='utf-8'),encoding='utf-8')
            (root/'data/products.csv').write_text('product_id,name,category,price\nP9,테스트 코트,아우터,22000\nP10,테스트 상의,상의,\n',encoding='utf-8')
            (root/'data/recommendation_metrics.json').write_text(json.dumps({'hit_rate_at_50':0.25}),encoding='utf-8')
            html,source=storefront_html(root)
            self.assertEqual(source,'csv')
            self.assertIn('테스트 코트',html)
            self.assertIn('"price": null',html)
            self.assertIn('<label>추천 HitRate@50</label><strong>0.2500</strong>',html)
            self.assertIn('<label>BM25 전체 MRR@10</label><strong>미제공</strong>',html)


if __name__=='__main__': unittest.main()
