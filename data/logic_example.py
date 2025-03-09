LOGIC_MRC_COT_4_SHOT = """
Example 1:

Passage: A professional baseball team manager, in order to have the funds to sign a new second-baseman, discreetly arranged to trade one of the most popular outfielders on the team for a lesser-known player and an undisclosed amount of money. The manager secretly considered the outfielder to be overrated and overpaid. Reporters forcefully criticized the trade, arguing that the team had lost its best player and that the manager had violated his fiduciary duties to the team and the fans. A few weeks after being traded, the outfielder was retraded, for twice the value received by the original team manager. Plainly, the outfielder' s price shows that the reporters' criticism of the manager' s action was accurate.
Question: The reasoning in the argument is vulnerable to the criticism that the argument does which one of the following?
A. The argument bases its conclusion on what the best decision is for the present on uncertain projections about what the best decision will be for the future.
B. The argument rejects a well-established way of achieving an end without explaining why an unconventional way is better suited for achieving the end.
C. The argument ignores the opinions of expert reporters in the field of baseball when there is no superior source of information.
D. The argument bases its conclusion on facts that could, considering the circumstances, have been the product of circumstances other than those presumed by the argument's proponents.

The reasoning steps are:

The passage argues that the reporters' criticism of the manager's decision was accurate because the outfielder was retraded for twice the value. 
However, this change in the outfielder's value could be due to various factors other than the manager's poor decision-making, such as market fluctuations, the outfielder's performance, or the demands of other teams. 
The argument does not consider these other possible circumstances when concluding that the manager's decision was wrong.
The answer is D.<end_of_reasoning>


Example 2:

Passage: Quality control investigator: Upon testing samples of products from our supplier that were sent by our field inspectors from various manufacturing locations, our laboratory discovered that over 20 percent of the samples were defective. Since our supplier is contractually required to limit the rate of defects among items it manufactures for us to below 5 percent, it has violated its contract with us.
Question: The reasoning in the quality control investigator's argument is flawed in that the argument
A. presumes, without providing justification, that the field inspectors were just as likely to choose a defective item for testing as they were to choose a nondefective item
B. presumes, without providing justification, that the field inspectors made an equal number of visits to each of the various manufacturing sites of the supplier
C. overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective
D. bases its conclusion on too small a sample of items tested by the laboratory

The reasoning steps are:

The reasoning in the quality control investigator's argument is flawed because it overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective.
This means that the 20 percent defect rate discovered by the laboratory might not accurately represent the overall defect rate among the items manufactured by the supplier.
The answer is: C.<end_of_reasoning>


Example 3:

Passage: The Levant -- the area that borders the eastern Mediterranean-was heavily populated in prehistoric times. The southern Levant was abandoned about 6, 000 years ago, although the northern Levant, which shared the same climate, remained heavily populated. Recently archaeologists have hypothesized that the sudden depopulation in the southern Levant was due to an economic collapse resulting from deforestation.
Question: If the statements above are true and the archaeologists' hypothesis is correct, which one of the following CANNOT be true?
A. The sheep and goats herded by the peoples of the southern Levant until 6, 000 years ago grazed extensively on the seedlings and saplings of indigenous tree species.
B. Carbon dating of organic remains from the southern Levant reliably demonstrates that there were no forests present in that area prior to 6, 000 years ago.
C. Organic remains from the northern Levant reliably indicate that tree species flourished there without interruption during the period when the southern Levant was being abandoned.
D. Since there are few traces of either quarried stone or of mud brick in buildings excavated in the southern Levant, it is likely that the buildings built there prior to 6, 000 years ago were made almost entirely of timber.

The reasoning steps are:

If there were no forests present in the southern Levant prior to 6, 000 years ago, it would not make sense for the cause of the economic collapse to be deforestation, as the area would have already been deforested. 
This contradicts the archaeologists' hypothesis, making option B the correct answer."
The answer is: B.<end_of_reasoning>


Example 4:

Passage: The most successful economies have been, and will continue to be, those that train as many people as possible in the human skills required to research, to develop, and to apply new technology. Japan is a model for this sort of training effort. Europe as a whole is in a weaker position: there is a shortage of skilled labor trained to use the new technologies, and there are not enough scientists able to develop and apply the technology. However, even in Japan there is a shortage of technically qualified people, and, like most European countries, Japan has far too many workers qualified to perform only menial tasks.
Question: Which one of the following can be properly inferred from the passage?
A. To be economically more successful, Europe needs to train more people in the new technologies.
B. Japan's successful economy depends upon an uncommonly narrow base of highly skilled labor.
C. Japan is not the best country against which to measure a country's economic success.
D. European countries have economies that are more successful than those of most other countries.

The reasoning steps are:

To be economically more successful, Europe needs to train more people in the new technologies.
The passage states that the most successful economies are those that train as many people as possible in the required human skills.
It also mentions that Europe is in a weaker position due to a shortage of skilled labor and scientists in new technologies.
Therefore, we can infer that training more people in new technologies would help Europe to become economically more successful.
The answer is A.<end_of_reasoning>
""".strip()

LOGIC_MRC_4_SHOT_WO_COT = """
Example 1:

Passage: A professional baseball team manager, in order to have the funds to sign a new second-baseman, discreetly arranged to trade one of the most popular outfielders on the team for a lesser-known player and an undisclosed amount of money. The manager secretly considered the outfielder to be overrated and overpaid. Reporters forcefully criticized the trade, arguing that the team had lost its best player and that the manager had violated his fiduciary duties to the team and the fans. A few weeks after being traded, the outfielder was retraded, for twice the value received by the original team manager. Plainly, the outfielder' s price shows that the reporters' criticism of the manager' s action was accurate.
Question: The reasoning in the argument is vulnerable to the criticism that the argument does which one of the following?
A. The argument bases its conclusion on what the best decision is for the present on uncertain projections about what the best decision will be for the future.
B. The argument rejects a well-established way of achieving an end without explaining why an unconventional way is better suited for achieving the end.
C. The argument ignores the opinions of expert reporters in the field of baseball when there is no superior source of information.
D. The argument bases its conclusion on facts that could, considering the circumstances, have been the product of circumstances other than those presumed by the argument's proponents.
The answer is D.<end_of_reasoning>


Example 2:

Passage: Quality control investigator: Upon testing samples of products from our supplier that were sent by our field inspectors from various manufacturing locations, our laboratory discovered that over 20 percent of the samples were defective. Since our supplier is contractually required to limit the rate of defects among items it manufactures for us to below 5 percent, it has violated its contract with us.
Question: The reasoning in the quality control investigator's argument is flawed in that the argument
A. presumes, without providing justification, that the field inspectors were just as likely to choose a defective item for testing as they were to choose a nondefective item
B. presumes, without providing justification, that the field inspectors made an equal number of visits to each of the various manufacturing sites of the supplier
C. overlooks the possibility that the field inspectors tend to choose items for testing that they suspect are defective
D. bases its conclusion on too small a sample of items tested by the laboratory
The answer is: C.<end_of_reasoning>


Example 3:

Passage: The Levant -- the area that borders the eastern Mediterranean-was heavily populated in prehistoric times. The southern Levant was abandoned about 6, 000 years ago, although the northern Levant, which shared the same climate, remained heavily populated. Recently archaeologists have hypothesized that the sudden depopulation in the southern Levant was due to an economic collapse resulting from deforestation.
Question: If the statements above are true and the archaeologists' hypothesis is correct, which one of the following CANNOT be true?
A. The sheep and goats herded by the peoples of the southern Levant until 6, 000 years ago grazed extensively on the seedlings and saplings of indigenous tree species.
B. Carbon dating of organic remains from the southern Levant reliably demonstrates that there were no forests present in that area prior to 6, 000 years ago.
C. Organic remains from the northern Levant reliably indicate that tree species flourished there without interruption during the period when the southern Levant was being abandoned.
D. Since there are few traces of either quarried stone or of mud brick in buildings excavated in the southern Levant, it is likely that the buildings built there prior to 6, 000 years ago were made almost entirely of timber.
The answer is: B.<end_of_reasoning>


Example 4:

Passage: The most successful economies have been, and will continue to be, those that train as many people as possible in the human skills required to research, to develop, and to apply new technology. Japan is a model for this sort of training effort. Europe as a whole is in a weaker position: there is a shortage of skilled labor trained to use the new technologies, and there are not enough scientists able to develop and apply the technology. However, even in Japan there is a shortage of technically qualified people, and, like most European countries, Japan has far too many workers qualified to perform only menial tasks.
Question: Which one of the following can be properly inferred from the passage?
A. To be economically more successful, Europe needs to train more people in the new technologies.
B. Japan's successful economy depends upon an uncommonly narrow base of highly skilled labor.
C. Japan is not the best country against which to measure a country's economic success.
D. European countries have economies that are more successful than those of most other countries.
he answer is A.<end_of_reasoning>
""".strip()