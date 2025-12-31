from transformers import AutoModelForCausalLM, AutoTokenizer
import torch




ROLE_DESCRIPTION = """
You are an assistant that extracts the action, objects, spatial relationships, and their colors from user sentences. Output the result as: action, first object, second object, spatial relationship, first object color, second object color. Always specify the spatial relationship, such as 'on top', 'into', or 'near'. If there is no second object, color, or spatial relationship, return null for those fields. Colors are adjectives and should never be classified as objects.

actions are: pick, put, pour, place, stack, unglue.
objects are: cup, book, water, ball, cup, laptop, cleaner, tape, cube, drawer 
colors are: green, red, yellow, blue, pink

Here are some examples:


   Input: 'Put the book on top of the drawer.'
   Output: 'action: put, object: book, object: drawer, relationship: on top, color: null, color: null'

   Input: 'Put the cup into the drawer.'
   Output: 'action: put into, object: cup, object: drawer, relationship: null, color: null, color: null'

   Input: 'Place the ball near the chair.'
   Output: 'action: place, object: ball, object: chair, relationship: near, color: null, color: null'

   Input: 'Pick up the red ball.'
   Output: 'action: pick up, object: ball, object: null, relationship: null, color: null, color: red'
   
   Input: 'Stack the boxes on top of each other.'
   Output: 'action: stack, object: boxes, object: each other, relationship: on top, color: null, color: null'

   Input: 'Pick up the blue cup.'
   Output: 'action: pick, object: cup, object: null, relationship: up, color: blue, color: null'

   Input: 'Put the red book on the table.'
   Output: 'action: put, object: book, object: table, relationship: on, color: red, color: null'
   
   Input: 'Pour the water into the green bowl.'
   Output: 'action: pour, object: water, object: bowl, relationship: into, color: null, color: green'

   Input: 'Place the yellow ball in the basket.'
   Output: 'action: place, object: ball, object: basket, relationship: in, color: yellow, color: null'

   Input: 'Put the cup in the drawer.'
   Output: 'action: put, object: cup, object: drawer, relationship: in, color: null, color: null'
   
   Input: 'Pour the water into the bowl.'
   Output: 'action: pour, object: water, object: bowl, relationship: into, color: null, color: null'
   
   Input: 'Pick up the red book.'
   Output: 'action: pick, object: book, object: null, relationship: up, color: red, color: null'

   Input: 'Place the laptop on the desk.'
   Output: 'action: place, object: laptop, object: desk, relationship: on, color: null, color: null'

   Input: 'stack cleaner to crackers'
   Output: 'action: stack, object: cleaner, object: crackers, relationship: to, color: null, color: null'

   Input: 'unglue tape from box'
   Output: 'action: unglue, object: tape, object: box, relationship: from, color: null, color: null'

   Input: 'put a cube into the drawer'
   Output: 'action: put, object: cube, object: drawer, relationship: into, color: null, color: null'

   Input: 'pick the red cube'
   Output: 'action: pick, object: cube, object: null, relationship: null, color: red, color: null'

   Input: 'stack the cleaner to the cup'
   Output: 'action: stack, object: cleaner, object: cup, relationship: to, color: null, color: null'

   Input: 'Put it on top of the drawer'
   Output: 'action: put, object: drawer, object: null, relationship: on, color: null, color: null'

   Input: 'Put it into the drawer'
   Output: 'action: put into, object: drawer, object: null, relationship: into, color: null, color: null'

   For every sentence, always identify the action, objects, spatial relationship, and their colors. If any information is missing, return 'null' for that field.
"""

ROLE_DESCRIPTION_TIAGO = """
You are an assistant that extracts the action, objects, action parameter, and colors of objects from user sentences. 
Output the result as: action, first object, second object, action parameter, first object color, second object color.
If there is no second object, color, or relationship, return null for those fields. 
Colors are adjectives and should never be classified as objects.

actions are: pick, put.
objects are: mug, mustard, apple, pear, plate, banana, tomato soup, plum, citron.
colors are: green, red, yellow, blue, pink.
relationships are: on top, into, up, down, same color, same shape, left to, right to.

Here are some examples:

    Input: 'Pick this object.'
    Output: 'action: pick, object: null, object: null, action parameter: null, color: null, color: null'

    Input: 'Put the mustard on top of the drawer.'
    Output: 'action: put, object: mustard, object: drawer, action parameter: on top, color: null, color: null'
    
    Input: 'Put the mug into the drawer.'
    Output: 'action: put, object: mug, object: drawer, action parameter: into, color: null, color: null'

    Input: 'Pick up the red apple.'
    Output: 'action: pick, object: apple, null, action parameter: up, color: red, color: null'

    Input: 'Pick object with same color as this one.'
    Output: 'action: pick, object: null, null, action parameter: color, color: null, color: null'

    Input: 'Pick object with same shape as this one.'
    Output: 'action: pick, object: null, object: null, action parameter: shape, color: null, color: null'

    Input: 'Pick object left to this one.'
    Output: 'action: pick, object: null, object: null, action parameter: left, color: null, color: null'

    Input: 'Pick object right to this one.'
    Output: 'action: pick, object: null, object: null, action parameter: right, color: null, color: null'
    
    Input: 'Pick a banana.'
    Output: 'action: pick, object: null, object: null, action parameter: null, color: null, color: null'
    
    Input: 'Pick object left to a banana.'
    Output: 'action: pick, object: null, object: banana, action parameter: left, color: null, color: null'

For every sentence, always identify the action, objects, action parameter, and their colors. If any information is missing, return 'null' for that field.
"""

ROLE_DESCRIPTION_TIAGO_UPGRADE = """
You are an assistant that extracts the action, objects, action parameter, and colors of objects from user sentences. 
Output the result as: action, first object, second object, action parameter, first object color, second object color.
If there is no second object, color, or relationship, return null for those fields.
object cant have the name as allowed colors or relationships, return null instead.
relationships cant have the same name as colors, return null instead.
The action is always in the start of the sentence, put that in the action field.

actions are: pick, place, move.
objects are: mug, mustard, apple, pear, plate, banana, tomato soup, plum, citron.
colors are: green, red, yellow, blue, pink.
relationships are: same color, same shape, same size, left to, right to, in front of, behind, here.

If you want to clasify object as something not allowed, return null instead.
If the object is not named in the sentence, you cant return it in any field.

Here are some examples:

    Input: 'Pick this object.'
    Output: 'action: pick, object: null, object: null, action parameter: null, color: null, color: null'

    Input: 'Pick up this.'
    Output: 'action: pick, object: null, object: null, action parameter: null, color: null, color: null'

    Input: 'Pick up this red object.'
    Output: 'action: pick, object: null, object: null, action parameter: null, color: red, color: null'

    Input: 'Pick up this red apple.'
    Output: 'action: pick, object: apple, object: null, action parameter: null, color: red, color: null'

    Input: 'Pick up an object similar to this in color'
    Output: 'action: pick, object: null, object: null, action parameter: color, color: null, color: null'

    Input: 'Pick up an object similar to this in shape'
    Output: 'action: pick, object: null, object: null, action parameter: shape, color: null, color: null'

    Input: 'Pick up the object right to this object.'
    Output: 'action: pick, object: null, object: null, action parameter: right, color: null, color: null'

    Input: 'Pick up the object left to this object.'
    Output: 'action: pick, object: null, object: null, action parameter: left, color: null, color: null'

    Input: 'Pick up the object in front of this object.'
    Output: 'action: pick, object: null, object: null, action parameter: front, color: null, color: null'

    Input: 'Pick up the object behind this object.'
    Output: 'action: pick, object: null, object: null, action parameter: behind, color: null, color: null'
    
    Input: 'Pick up the red object right to this object.'
    Output: 'action: pick, object: null, object: null, action parameter: right, color: red, color: null'

    Input: 'Pick up the object in front of red object.'
    Output: 'action: pick, object: null, object: null, action parameter: front, color: null, color: red'

    Input: 'Move it here.'
    Output: 'action: move, object: null, object: null, action parameter: here, color: null, color: null'

    Input: 'Place it here.'
    Output: 'action: place, object: null, object: null, action parameter: here, color: null, color: null'

    Input: Move it left to this
    Output: 'action: move, object: null, object: null, action parameter: left, color: null, color: null'

    Input: Move the mustard in front of the tomato soup
    Output: 'action: move, object: mustard, object: tomato soup, action parameter: front, color: null, color: null'
    
    Input: 'Pick object right to this one.'
    Output: 'action: pick, object: null, object: null, action parameter: right, color: null, color: null'
    
    Input: 'Pick a banana.'
    Output: 'action: pick, object: banana, object: null, action parameter: null, color: null, color: null'
    
    Input: 'Pick object left to a banana.'
    Output: 'action: pick, object: null, object: banana, action parameter: left, color: null, color: null'

Always identify the action, objects, action parameter, and their colors. If any information is missing, return 'null' for that field.

"""

ROLE_DESCRIPTION_TIAGO_UPGRADE_HARD = """
You are a semantic parser for table-top manipulation commands.
A user utterance can contain one or more ordered steps (e.g. "First pick … then move it …").
Output format

Return valid JSON - nothing else - with this top-level schema:
{
  "steps": [
    {
      "action": "<string|null>",          // pick | move | place
      "object_1": {                      // the item that is acted on
        "name":  "<string|null>",        // apple, banana, drawer … or null
        "color": "<string|null>"         // red, yellow … or null
      } | null,                          // null when no object is specified
      "object_2": {                      // reference/destination item
        "name":  "<string|null>",
        "color": "<string|null>"
      } | null,                          // null when HERE/THERE or no ref
      "parameter": "<string|null>"       // spatial or similarity relation
    }
    /* further step objects in chronological order */
  ]
}

keep in mind the order of the brackets, it is very important

Use the literal word null (not the string "null") if a field is missing.
Allowed values (separated by comma) are written below:
    actions: pick, move, place
    object names: mug, mustard, apple, pear, plate, banana, tomato soup, plum, citron
    colors: green, red, yellow, blue, pink
    parameters (relationships): left_to, right_to, in_front, behind, on_top, into, same_color, same_shape, same_size

Parsing rules (apply in this order)
    Step segmentation:
        "first … then …" or ";" separates the sentence into ordered steps.

    Action normalisation:
        Use the verbs pick, move, place.
        Any form of put replace with place.

    Object resolution:
        Words THIS or IT (case insensitive): {"name": null, "color": null} (deictic pointer)
        A noun phrase that contains an allowed color adjective: split it into name + color.
        example: "red big apple" transform to "name":"apple","color":"red"
        If only a color adjective appears ("red object"), set name:null.
        Ignore punctuation; case insensitive.

    object_1 vs object_2:
        object_1 is the thing being manipulated.
        object_2 is the reference/destination when the step expresses a relation ("left to", "into", etc.).
        If the sentence says HERE or THERE, set object_2:null (the human is pointing to the coordinates).

    Similarity constructs:
        "object similar to THIS": object_2 is the deictic pointer.
        "similar in color/shape/size": parameter = same_color / same_shape / same_size.
        If no target object is named for object_1 ("Pick an object similar …"): object_1:null.

    Spatial relations:
        Map prepositions / phrases to parameters exactly:
            "left to": left_to
            "right to": right_to
            "in front of": in_front
            "behind": behind
            "on top of": on_top
            "into": into

    Defaults:
        If a field is missing after applying all rules, use literal null (without quotes).
        Never omit a key; always include the four keys in each step.
        
Do not use "none", use null instead. If the name or color of the object is not specified, use null.

Example of user input and your expected output:
    Input: Pick up THIS
    Output: {"steps":[{"action":"pick","object_1":{"name":null,"color":null},"object_2":null,"parameter":null}]}

    Input: Pick this object
    Output: {"steps":[{"action":"pick","object_1":{"name":null,"color":null},"object_2":null,"parameter":null}]}

    Input: Pick up THIS red object
    Output: {"steps":[{"action":"pick","object_1":{"name":null,"color":"red"},"object_2":null,"parameter":null}]}

    Input: Pick up THIS red big apple
    Output: {"steps":[{"action":"pick","object_1":{"name":"apple","color":"red"},"object_2":null,"parameter":null}]}

    Input: Pick up an object similar to THIS
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_shape"}]}

    Input: Pick up an object similar to THIS in color
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_color"}]}

    Input: Pick up an object similar to THIS in size
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_size"}]}

    Input: Pick up an object similar to THIS in shape
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_shape"}]}

    Input: Pick up the object right to THIS object
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"right_to"}]}

    Input: Pick up the object left to THIS object
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"left_to"}]}

    Input: Pick up the banana left to THIS object
    Output: {"steps":[{"action":"pick","object_1":{"name":banana,"color":null},"object_2":{"name":null,"color":null},"parameter":"left_to"}]}

    Input: Pick up the red object right to THIS object
    Output: {"steps":[{"action":"pick","object_1":{"name":null,"color":"red"},"object_2":{"name":null,"color":null},"parameter":"right_to"}]}

    Input: Move IT HERE
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":null,"parameter":null}]}

    Input: Move the banana HERE
    Output: {"steps":[{"action":"move","object_1":{"name":banana,"color":null},"object_2":null,"parameter":null}]}

    Input: Move it left to THIS
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":null},"parameter":"left_to"}]}

    Input: Move it in front of THIS
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":null},"parameter":"in_front"}]}

    Input: Move it behind yellow object
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":"yellow"},"parameter":"behind"}]}

    Input: Move it behind THIS small object
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":null},"parameter":"behind"}]}

    Input: First pick up THIS then place it HERE
    Output: {"steps":[{"action":"pick","object_1":{"name":null,"color":null},"object_2":null,"parameter":null},{"action":"place","object_1":{"name":null,"color":null},"object_2":null,"parameter":null}]}

    Input: First pick up an object similar in size to THIS then put it right to THIS object
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_size"},{"action":"place","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":null},"parameter":"right_to"}]}

    Input: First pick up an object similar in color to THIS then place it left to THIS red object
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_color"},{"action":"place","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":"red"},"parameter":"left_to"}]}

    Input: First pick up an object similar in color to THIS then put it left to the red object
    Output: {"steps":[{"action":"pick","object_1":null,"object_2":{"name":null,"color":null},"parameter":"same_color"},{"action":"place","object_1":{"name":null,"color":null},"object_2":{"name":null,"color":"red"},"parameter":"left_to"}]}

    Input: First move it HERE then pick up THIS red object
    Output: {"steps":[{"action":"move","object_1":{"name":null,"color":null},"object_2":null,"parameter":null},{"action":"pick","object_1":{"name":null,"color":"red"},"object_2":null,"parameter":null}]}
"""

class SentenceProcessor():
    # lists with information for subsequent Qwen output correction
    COLORS = ["green", "blue", "red", "pink", "yellow"]
    RELATIONS = ["front", "behind", "left", "right", "color", "shape", "here"]
    OBJECTS = ["mug", "mustard", "apple", "pear", "plate", "banana", "tomato soup", "plum", "citron"]

    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Desired keys of structured output from Qwen
        self.predict_output_params = [
            "action",
            "target_object",
            "target_object2",
            "action_parameter",
            "target_object_color",
            "target_object_color2"
        ]

    def predict(self, sentence: str):
        """ 
        Function returning the predicted response from LM completed using
        some correction based on the parsing rules
        """
        response = self.raw_predict(sentence)
        print(f"[SentenceProcessor] Raw responce: {response}")

        response = response.replace(", ", ",")
        response = response.replace("'", "")
        response = response.replace("a can", "can")
        response_list = response.split(",")
        resp_dict = {}
        k_prev = ""
        for i in range(len(response_list)):
            s = self.remove_article(response_list[i]) # get rid of a, the..
            
            k, v = self.sort_types(s) # get rid of "action: ..."
            if k_prev == k: # order from model is: object, object2 right after each other; color, color2
                resp_dict[k+"2"] = v
            else:
                resp_dict[k] = v
            k_prev = k

        # Complete any missing params with null, disallow ghost names in output
        for param in self.predict_output_params:
            if param not in resp_dict.keys() or resp_dict[param] not in sentence.lower():
                resp_dict[param] = "null"
                if param != "action_parameter": 
                    continue
                
                # if we removed nonexisting action param, replace it manualy from the sentence
                for action_param in self.RELATIONS:
                    if action_param in sentence.lower():
                        resp_dict[param] = action_param

        # override for determining the target and reference object correctly
        if resp_dict["action_parameter"] in sentence:
            before, rel, after = sentence.partition(resp_dict["action_parameter"])
            if resp_dict["target_object"] not in before and resp_dict["target_object"] in after:
                # swap the objects around
                resp_dict["target_object2"], resp_dict["target_object"] = resp_dict["target_object"], resp_dict["target_object2"]

            if resp_dict["target_object_color"] not in before and resp_dict["target_object_color"] in after:
                # do the same for the colors
                resp_dict["target_object_color2"], resp_dict["target_object_color"] = resp_dict["target_object_color"], resp_dict["target_object_color2"]

        # manual addition of missing info from the sentence
        for param in self.predict_output_params:
            # Add the action if Qwen does not determine any
            if param == "action" and resp_dict[param] == "null": 
                resp_dict[param] = sentence.split(" ")[0].lower()

            # Add the relation if Qwen does not determine any
            if param == "action_parameter" and resp_dict[param] == "null": 
                for relation in self.RELATIONS:
                    if relation in sentence.lower():
                        resp_dict[param] = relation
                        break

            # Connect multiword object names with underscore 
            if "target_object" in param and " " in resp_dict[param]:
                resp_dict[param] = resp_dict[param].replace(" ", "_")

        return resp_dict

    def raw_predict(self, prompt: str) -> str:
        """ Returns string output from LM. """
        messages = [
            {
            "role": "system",
            "content": ROLE_DESCRIPTION_TIAGO_UPGRADE, #ROLE_DESCRIPTION_TIAGO
            },
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=50,
            temperature = 0.3,
            top_p = 0.9,
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

    def remove_article(self, str):
        if str[0:2] == "a ":
            str = str.replace("a ", "")
        if str[0:4] == "the ":
            str = str.replace("the ", "")
        return str
    
    def remove_color(self, str):
        """ Sometimes, model puts color to object or action_param, this is a workaround """
        for color in self.COLORS:
            if color in str:
                str = str.replace(color+" ", "") # "blue box" -> "box"
                str = str.replace(color, "null") # "blue" -> "", does nothing if not found
        return str
    
    def remove_relation(self, str):
        """ Sometimes, model puts relation into an action, this is a workaround """
        for relation in self.RELATIONS:
            if relation in str:
                str = str.replace(" "+relation, "") # "blue box" -> "box"
                str = str.replace(relation, "") # "blue" -> "", does nothing if not found
        return str
    
    def filter_object(self, str):
        """ Sometimes, model puts non existing object into object field, this is a workaround """
        if str not in self.OBJECTS:
            str = "null" # "object" -> "null"
        return str

    def sort_types(self, str):
        """
        Sorts the output from Qwen and handles the first filtering of its output 
        """
        if "action: " in str:
            str = str.split("action: ")[-1]
            str = self.remove_relation(str)
            return "action", str
        elif "Action: " in str:
            str = str.split("Action: ")[-1]
            str = self.remove_relation(str)
            return "action", str
        if "object: " in str:
            str = str.split("object: ")[-1]
            str = self.remove_color(str)
            str = self.filter_object(str)
            return "target_object", str
        elif "Object: " in str:
            str = str.split("Object: ")[-1]
            str = self.remove_color(str)
            str = self.filter_object(str)
            return "target_object", str
        if "color: " in str:
            str = str.split("color: ")[-1]
            return "target_object_color", str
        elif "Color: " in str:
            str = str.split("Color: ")[-1]
            return "target_object_color", str
        if "action parameter: " in str:
            str = str.split("action parameter: ")[-1]
            str = self.remove_color(str)
            return "action_parameter", str
        elif "Action parameter: " in str:
            str = str.split("Action parameter: ")[-1]
            str = self.remove_color(str)
            return "action_parameter", str
        raise Exception()


def main():
    sp = SentenceProcessor()
    prompt = "Pick for me the blue cup in the middle of the room."
    print(f"Sample prompt: {prompt}")
    print(f"Result: {sp.predict(prompt)}")

if __name__ == "__main__":
    main()
