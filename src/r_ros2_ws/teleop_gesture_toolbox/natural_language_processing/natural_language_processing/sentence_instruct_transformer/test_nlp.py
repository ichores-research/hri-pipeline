
from sentence_processor import SentenceProcessor
    
# FORMAT EXAMPLE:
# "stack cleaner to crackers", # Input sentence 
# {'target_action': 'stack', 'target_object': "cleaner", 'target_object2': "crackers", "relationship": "to"} # Items checked for correct solution

def test_nlp_1(test_dataset = [
        [
            # Input sentence
            "stack cleaner to crackers", 
            # Items checked for correct solution
            {'target_action': 'stack', 'target_object': "cleaner", 'target_object2': "crackers", "relationship": "to"}
        ], [
            "stack cleaner to cube", 
            {'target_action': 'stack', 'target_object': "cleaner", 'target_object2': "cube", "relationship": "to"}
        ], [
            "stack the cleaner to the cup", 
            {'target_action': 'stack', 'target_object': "cleaner", 'target_object2': "cup", "relationship": "to"}
        ], [
            "stack a can to cleaner", 
            {'target_action': 'stack', 'target_object': "can", 'target_object2': "cleaner", "relationship": "to"}
        ], [
            "unglue a can", 
            {'target_action': 'unglue', 'target_object': "can",}
        ], [
            "push cube", 
            {'target_action': 'push', 'target_object': "cube",}
        ], [
            "pick cleaner",
            {'target_action': 'pick', 'target_object': "cleaner"}
        ], [
            "unglue box",
            {'target_action': 'unglue', 'target_object': "box"}
        ], [
            "push a can", 
            {"target_action": "push", "target_object": "can"}
        ], [
            "put cleaner into drawer", 
            {"target_action": "put", "target_object": "cleaner", "target_object2": "drawer", "relationship": "into"}
        ], [
            "unglue cleaner", 
            {"target_action": "unglue", "target_object": "cleaner"}
        ], [
            "pour a can to bowl", 
            {"target_action": "pour", "target_object": "can", "target_object2": "bowl"}
        ], [
            "stack cleaner to crackers",
            {"target_action": "stack", "target_object": "cleaner", "target_object2": "crackers", "relationship": "to"}
        ], [
            "put cube into the drawer",
            {"target_action": "put", "target_object": "cube", "target_object2":"drawer"}
        ], [
            "pick for me the green box",
            {"target_action": "pick", "target_object": "box", "target_object2":""}
        ], [
            "Please, pick me the pink cleaner.",
            {"target_action": "pick", "target_object": "cleaner", "target_object2":""}
        ], [
            "Please, stack the cleaner on top of the box.",
            {"target_action": "stack", "target_object": "cleaner", "target_object2":"box", "relationship": "on"}
        ],
        [
            "put it on top the drawer.",
            {"target_action": "put", "target_object": "item", "target_object2":"drawer", "relationship": "on top"}
        ], [
            "put it into the drawer.", 
            {"target_action": "put", "target_object": "item", "target_object2":"drawer", "relationship": "into"}
        ], [
            "pick same color.",
            {"target_action": "pick", "target_object": "null", "target_object2": "null", "relationship": "color"}
        ], [
            "pick same shape.",
            {"target_action": "pick", "target_object": "null", "target_object2": "null", "relationship": "shape"}
        ], [
            "pick object left to this.",
            {"target_action": "pick", "target_object": "null", "target_object2": "null", "relationship": "left"}
        ], [
            "pick object right to this.",
            {"target_action": "pick", "target_object": "null", "target_object2": "null", "relationship": "right"}
        ],
    ]):
    sp = SentenceProcessor()
    
    for sentence, solution in test_dataset:
        instruct_response = sp.predict(sentence)
        for key in solution.keys():
            assert instruct_response[key] == solution[key], f"sentence: {sentence}\n{key}: {instruct_response[key]} != {solution[key]}\nSolution: {instruct_response}"
    print("test successful")

def test_complex_sentences(test_dataset = [
        [
            "Hello, take that blue cube and put it on top the drawer.",
            {"target_action": "take", "target_object": "cube", "target_object2":"drawer"}
        ],
    ]):
    pass # TODO: 

if __name__ == '__main__':
    test_nlp_1()