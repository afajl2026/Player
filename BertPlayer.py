import random
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from Player import Player

class BERTPlayer(Player):
    PLAYER_NAME = "BERT Player"  # name for the player

    def __init__(self):
        super().__init__(self.PLAYER_NAME)
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=7)
        self.model_path = './finetuned_bert_foradj'
        self.model_finetune = BertForSequenceClassification.from_pretrained(self.model_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #load the model, load the pretrained model, initialize tokenizer.


    def get_Noun(self, target, hand):
        input_text = f"{target}[SEP]" + "[SEP] ".join(hand) + " [SEP]"
        inputs = self.tokenizer(input_text, return_tensors='pt')
        # padding=True, truncation=True, max_length=128
        # tokenize the input cuz later we need it in the model
        print(inputs)
        # Get model prediction

        with torch.no_grad():
            outputs = self.model(**inputs)
            # logits = outputs.logits
            logits = outputs.logits[:, :len(hand)]  # Only consider logits for the given choices
        print(f"Model outputs: {outputs}")
        # get prediction without doing gradient stuff
        probability = torch.nn.functional.softmax(logits,dim=1)
        print(f"Probabilities: {probability}")
        # get probability from each noun
        max_prob,max_index = torch.max(probability,dim=1)
        print(f"Max probability: {max_prob}, Max index: {max_index}")
        # find the one with the highest probability

        if max_index.item() >= len(hand):
            print("Error: max_index out of range.")
            return None

        best_noun = hand[max_index.item()]
        return best_noun
        # Return the best noun from the  hand

    def choose_card(self, target, hand):
        # Select the index of the card from the cards list that is closest to target
        firstpick = self.get_Noun(target, hand)
        # bestCardNdx = hand.index(firstpick) if firstpick in hand else 0
        if firstpick is None:
            print(f"Error: first pick index is None")
            return 0  # Fallback to a default card index
        bestCardNdx = hand.index(firstpick)
        return bestCardNdx
        # if best card not found go 0

    def judge_card(self, target, player_cards):
        # Use the BERT model to judge the best card
        print(f"Judging: target={target}, player_cards={player_cards}")
        judges_pick = self.get_Noun(target, player_cards)
        print(f"Judge picked: {judges_pick}")
        if judges_pick not in player_cards:
            print(f"Error: Judge's pick {judges_pick} is not in player cards {player_cards}")
            return player_cards[0]
        return judges_pick


    def process_results(self, result):
        # Handle results returned from server
        print("Result", result)

if __name__ == '__main__':
    player = BERTPlayer()
    player.run()