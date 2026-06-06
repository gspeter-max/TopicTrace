import hashlib
from documentIngestion.graphPersistence import generate_stable_entity_id

def test_generate_stable_entity_id_is_consistent():
    """This test checks if the same name always gets the same code."""
    name = "Artish Mahara"
    id1 = generate_stable_entity_id(name)
    id2 = generate_stable_entity_id(name)
    assert id1 == id2
    assert len(id1) == 12

def test_generate_stable_entity_id_ignores_case_and_space():
    """This test checks if extra spaces or big letters don't confuse the code maker."""
    name1 = "  Artish Mahara  "
    name2 = "artish mahara"
    assert generate_stable_entity_id(name1) == generate_stable_entity_id(name2)
