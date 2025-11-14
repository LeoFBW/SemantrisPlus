1. Separate HTML and Main file, let use arcade.html for the HTML, hence update the python codes accordingly
2. We will have a list of words in general_vocab.txt under assets folder of the same directory which stores words seaparated by line. Words are randomly selected from this list with checks to ensure the same word is not was not in the game before(can use index/word based dict to ensure such?)
3. Instead of eliiminating the most similar word, remember in the arcade mode. there will be a target word to be removed, and as long as it's within the first four similar word, it will get removed with a new target word been generated. Hence similarly, the order will be reordered based on last entered word similarity from most correlated in the bottom, to least on the top using AI LLM. The word between the top 4 and the target word will be removed shall the target word within top 4 (you remove 4 words if you land target on the most correleted, but only 1 if it's 4th correlated).
4. Add a score counter, based on how many words eliminated









It's amazing how far you pulled it off, however we have some features to improve

1. Make a much better UI, one current problem is that it display only a few words, it should display a wide range of word, and here's a rough way to see how many words to display minimum 5, = user_score/2, with a  increase to max 20 to miimc the difficulty increase. Similarly the UI should be much more innovative and cutting edge in turns of display, think something of professional. yet minimaistic, not something relic 2000s early like ours currently
2. Add a upward counting counter to see how far this session logged. 





change ui, i want not a board in a matrix, but a tower like strucutre, with associaiton ocrrlation after each user enter an input with a gradient for 4th height and below(to tell about this is how much they need to get correlate by)   + new added word dropped form the top 

###### 

1. This is how i want my UI to be, it should resemble this, you see the blue word, that's the target word. Once a target is entered, there's an animiation shifting the iteams to the correlation, which takes 0.5sec, and then another 0.25 sec will be nothing, and then 0.75 sec of stuffs popping based on the line. This is a ui change, game mechanics maintain the same.
2. Of course, we shall include a timer as well

1. Above code got issue, comapred ot the original semantris, the ordering is wrong as in our version we don;t have line of removal, and our whole ordering corrleation system to the top is completley opposite of how semantris does it, rmb as long as the target word hits top 4 corrleation(which semantris is done bottom to top meaning most correlated word is at the bottom)let's fix this. Hence similarly, the order will be reordered based on last entered word similarity from most correlated in the bottom, to least on the top using AI LLM. The word between the top 4 and the target word will be removed shall the target word within top 4 (you remove 4 words if you land target on the most correleted, but only 1 if it's 4th correlated).
2. Also improve the UI, to be more professional, minimalsitic, yet user firendly and simialr to the original sematnris by google 7 yr old ago







Still a few main problems

1.  Word correlations: most corrleated gone to the top of twoer, in the semantris, it's at the bottom(most correalted) to top(least correalted)
2. Dispaly of tower should have space suffice for 21, we got  12 word space vertically, fix please, a lot of the screen is empty vertically speaking





Thank you, we fix all major issues, however still some more needed not realted to the mechanics, but display

1. animation and transition, ours is like sudden reordering and dropping new ones, In the real sematnris arcade mode, there's smooth trasnitions for reordering , such that the worsd swap place have a synced movement of traveling to destination at pace, and simiarly with annihilating words breaking apart(destorying and exploding), then words above it fall down to cover the annihilated height.   Simiarly new words are dropped from above with speed, instead of teleport like we are having now, fix this.
2.  in the real semantric arcade mode, there's an artifical line at the 4th height signaling line of annihilation. simiarly i want this.





**Readme.md:**

write a readme.md about this project saying feeling like the concept ofsemantris is amaizng, learn woridng, un, energyiv, however it's kinda depracted considering the whole AI LLM race we are currently in, with AI LLM booming since the introudciton of chatgpt 3.5 in late 2022, hence this is a version simiarly in nature, but use LLM's spirit(in fact Google's gemini) to acheive same purpose but more clever. stronger and more up to date, using the mountains knoweldge that modern LLM has bene entrusted with. We included quite a bit of text file to test out for you, which you can just search '.txt' in app.py and reaplce it with any other, or your own text file samples under the asset subfolder. 

Use requirements.txt to isntall necessry lib, also setup .env:

GEMINI_API_KEY = "ABCDEFG_API_KEY"

FLASK_SECRET_KEY=" 'ABCDEFG_SECRET_KEY'"



Enjoy the game!

Weakenss and drawback of using LLM, of course it's not detemrinistic, it's quite problaistic even with a tep of 0, but for most short pharses this is a small issue.



project issues: 

Animation stll many developement,

UI kinda suck, but core feature suffice

Needs an ending condition, right now it's kinda ended, which left long term TBD, but short term wise it would be who take shortest time to finish the same word list sample(that's why we included both timer and score lol).



Pull requirests welcome in every front!

this is a small demo only, please help your self configuering the API key which free tier should sufficee using gogole's AI studio API key, it's nowhere near smantirs groundbreaking in transfromatic deisgn arch or UI as i only spent an afternoon working on this proejct. Even iwth today's ;latest AI LLM, of course it's nowehere comparable to google's amaizng team of engineers acorss every stack. 



A plethora of AI LLM are used, hence i extend my huge commandation to them, they are changing many industries helping our lifestyles for the better, so bravo.



---



V0.04:

can you add a new mecanics, that regardless of hit or not:



The session['board'] is updated to be the ranked_list returned by the AI. This "shuffles" the board into the order the AI just created, serving as a minor board change, before the



Update Score: score is increased by the number of words removed.

Calculate New Board:

The words_removed are taken out of the current_board.

A new desired_size is calculated using calculate_board_size(new_score). The board grows as the score increases.

get_new_words() is called to add enough new words to meet this desired_size.

Set New Target: A new target_word is chosen from the newly added words.

Update Session: The board, score, and target_word in the session are all updated.

Send Response: The server sends a JSON response with hit: True, the new_board, new_target, and new_score.



logic commence as usual,



also



get_new_words() is called to add enough new words to meet this desired_size.





There is a system thoughg dict with key = word, preventing words added to the board b4 is been added again. hence there's another condition, if no target words is remaining, then the game freezes in time and point, as it;s an winning condition alr