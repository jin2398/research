import csv
import tempfile
import unittest
from pathlib import Path
from src.dashboard.catalog import demo_products, filter_products, load_products, suggest_products, image_for
from src.dashboard.api_adapter import parse_results


class CatalogTests(unittest.TestCase):
    def test_demo_search_and_sort(self):
        products=demo_products()
        self.assertEqual(len(products),7)
        coat=filter_products(products,'베이지 코트 코디')
        self.assertEqual(coat[0]['id'],'demo-001')
        self.assertEqual([p['id'] for p in filter_products(products,'아우터')][:2],['demo-001','demo-002'])
        sorted_products=filter_products(products,'',order='낮은 가격순')
        self.assertEqual(sorted_products[0]['id'],'demo-007')
        self.assertEqual(sorted_products[-1]['id'],'demo-001')

    def test_categories_and_styles(self):
        products=demo_products()
        self.assertTrue(all(p['category']=='아우터' for p in filter_products(products,category='아우터')))
        self.assertEqual(suggest_products(products,'페미닌')[0]['id'],'demo-005')
        for p in products:
            self.assertTrue(Path(image_for(p)).is_file())

    def test_csv_contract_and_path_safety(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'data').mkdir()
            products,mode=load_products(root)
            self.assertEqual(mode,'demo')
            path=root/'data/products.csv'
            with path.open('w',encoding='utf-8',newline='') as fp:
                out=csv.DictWriter(fp,fieldnames=['product_id','name','category','price','image_path'])
                out.writeheader()
                out.writerow({'product_id':'a','name':'실제 입력 예시','category':'상의','price':'12,000','image_path':'../../etc/passwd'})
            data,mode=load_products(root)
            self.assertEqual(mode,'csv')
            self.assertEqual(data[0]['price'],12000)
            self.assertIsNone(data[0]['image'])
            path.write_text('id,title\n1,foo\n',encoding='utf-8')
            with self.assertRaises(ValueError):load_products(root)

    def test_api_json_parser(self):
        self.assertEqual(parse_results({'results':[{'product_id':'1','name':'원피스','price':'90000'}]})[0]['price'],90000)
        with self.assertRaises(ValueError):parse_results({'message':'not the expected schema'})

if __name__=='__main__':unittest.main()
