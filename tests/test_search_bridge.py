"""Search integration contract tests that do not download CLIP model weights."""
from __future__ import annotations
import io
from PIL import Image
from fastapi.testclient import TestClient
import src.serving.search_bridge as bridge

client=TestClient(bridge.app)


def test_status_shape_with_or_without_team_zip_files():
    result=client.get('/api/search/status')
    assert result.status_code==200
    assert isinstance(result.json()['missing'],list)
    assert isinstance(result.json()['index_ready'],bool)
    if result.json()['index_ready']:
        assert not result.json()['missing']


def test_text_search_contract(monkeypatch):
    seen={}
    def mock_search(*,query='',image=None,top_k=20):
        seen.update(query=query,image=image,top_k=top_k)
        return {'results':[{'product_id':'TEST-001'}], 'search_type':'text'}
    monkeypatch.setattr(bridge,'execute_search',mock_search)
    response=client.post('/api/search',json={'query':'베이지 코트','mode':'text','top_k':7})
    assert response.status_code==200
    assert seen=={'query':'베이지 코트','image':None,'top_k':7}


def test_image_search_contract(monkeypatch):
    seen={}
    def mock_search(*,query='',image=None,top_k=20):
        seen.update(query=query,image=image and image.is_file(),top_k=top_k)
        return {'results':[],'search_type':'image'}
    monkeypatch.setattr(bridge,'execute_search',mock_search)
    image=Image.new('RGB',(8,8),'white')
    buff=io.BytesIO();image.save(buff,format='PNG')
    response=client.post('/api/search',data={'mode':'image','top_k':'10'},files={'file':('query.png',buff.getvalue(),'image/png')})
    assert response.status_code==200
    assert seen=={'query':'','image':True,'top_k':10}


def test_hybrid_requires_text_and_image():
    response=client.post('/api/search',data={'mode':'hybrid','query':'coat','top_k':'10'})
    assert response.status_code==422
    assert 'file' in str(response.json())


def test_bad_image_rejected_without_loading_model():
    response=client.post('/api/search',data={'mode':'image'},files={'file':('bad.png',b'not image','image/png')})
    assert response.status_code==422


def test_untrusted_image_path_rejected():
    response=client.get('/api/search/images/../etc/passwd')
    assert response.status_code in (400,404)
