MATH_POT_FEW_SHOT = """
Example 1:
The question is : Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
The solution code is:
```python
def solution():
    '''Olivia has $23. She bought five bagels for $3 each. How much money does she have left?'''
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    money_left = money_initial - money_spent
    result = money_left
    return result
```

Example 2:
The question is: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
The solution code is:
```python
def solution():
    '''Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?'''
    golf_balls_initial = 58
    golf_balls_lost_tuesday = 23
    golf_balls_lost_wednesday = 2
    golf_balls_left = golf_balls_initial - golf_balls_lost_tuesday - golf_balls_lost_wednesday
    result = golf_balls_left
    return result
```

""".strip()


MATH_COT_FEW_SHOT = """
Example 1:
The question is : Sam memorized six more digits of pi than Carlos memorized. Mina memorized six times as many digits of pi as Carlos memorized. If Mina memorized 24 digits of pi, how many digits did Sam memorize?
The reasoning steps are:

Mina memorized 24 digits of pi, and it is given that Mina memorized six times as many digits of pi as Carlos memorized. So, let's represent the number of digits Carlos memorized as 'x'. We can write this as an equation: 6x = 24.
To find the value of 'x', we need to divide both sides of the equation by 6. This will give us the number of digits Carlos memorized: x = 24 / 6.
Now, we calculate the value of 'x': x = 4. This means Carlos memorized 4 digits of pi.
It's given that Sam memorized six more digits of pi than Carlos. So, to find the number of digits Sam memorized, we add 6 to the number of digits Carlos memorized: Sam's digits = x + 6.
Now, we calculate the number of digits Sam memorized: Sam's digits = 4 + 6 = 10.
The answer is: \\boxed{10}.<end_of_reasoning>

Example 2:
The question is: Lee mows one lawn and charges $33. Last week he mowed 16 lawns and three customers each gave him a $10 tip. How many dollars did Lee earn mowing lawns last week?
The reasoning steps are:

Calculate the total amount from mowing the lawns.\nLee charges $33 per lawn and mowed 16 lawns. \nTotal amount from mowing lawns = $33 * 16\nTotal amount from mowing lawns = $528
Calculate the total amount from the tips.\nThree customers each gave him a $10 tip.\nTotal amount from tips = $10 * 3\nTotal amount from tips = $30
Add the total amount from mowing lawns and the total amount from tips to find the total amount Lee earned.\nTotal amount earned = Total amount from mowing lawns + Total amount from tips\nTotal amount earned = $528 + $30\nTotal amount earned = $558
The answer is: \\boxed{558}.<end_of_reasoning>

""".strip()



GSM_COT_8_SHOT = """
Example 1:
The question is : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
The reasoning steps are:

There are 15 trees originally. Then there were 21 trees after some more were planted. 
So there must have been 21 - 15 = 6.
The answer is: \\boxed{6}.<end_of_reasoning>


Example 2:
The question is: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
The reasoning steps are:

There are originally 3 cars. 
2 more cars arrive. 
3 + 2 = 5.
The answer is: \\boxed{5}.<end_of_reasoning>


Example 3:
The question is: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
The reasoning steps are:

Originally, Leah had 32 chocolates. Her sister had 42. 
So in total they had 32 + 42 = 74. 
After eating 35, they had 74 - 35 = 39.
The answer is: \\boxed{39}.<end_of_reasoning>


Example 4:
The question is: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
The reasoning steps are:

Jason started with 20 lollipops.
Then he had 12 after giving some to Denny.
So he gave Denny 20 - 12 = 8.
The answer is: \\boxed{8}.<end_of_reasoning>


Example 5:
The question is: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
The reasoning steps are:

Shawn started with 5 toys. 
If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9. 
The answer is: \\boxed{9}.<end_of_reasoning>


Example 6:
The question is: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
The reasoning steps are:

There were originally 9 computers.
For each of 4 days, 5 more computers were added.
So 5 * 4 = 20 computers were added. 9 + 20 is 29.
The answer is: \\boxed{29}.<end_of_reasoning>


Example 7:
The question is: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?
The reasoning steps are:

Michael started with 58 golf balls.
After losing 23 on tuesday, he had 58 - 23 = 35.
After losing 2 more, he had 35 - 2 = 33 golf balls.
The answer is: \\boxed{33}.<end_of_reasoning>


Example 8:
The question is: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
The reasoning steps are:

Olivia had 23 dollars.
5 bagels for 3 dollars each will be 5 x 3 = 15 dollars.
So she has 23 - 15 dollars left. 23 - 15 is 8.
The answer is: \\boxed{8}.<end_of_reasoning>

""".strip()

GSM_COT_8_SHOT_WO_COT = """
Example 1:
The question is : There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

The answer is: \\boxed{6}.<end_of_reasoning>


Example 2:
The question is: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

The answer is: \\boxed{5}.<end_of_reasoning>


Example 3:
The question is: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

The answer is: \\boxed{39}.<end_of_reasoning>


Example 4:
The question is: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

The answer is: \\boxed{8}.<end_of_reasoning>


Example 5:
The question is: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?

The answer is: \\boxed{9}.<end_of_reasoning>


Example 6:
The question is: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?

The answer is: \\boxed{29}.<end_of_reasoning>


Example 7:
The question is: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

The answer is: \\boxed{33}.<end_of_reasoning>


Example 8:
The question is: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?

The answer is: \\boxed{8}.<end_of_reasoning>

""".strip()

MATH_COT_4_SHOT_WO_COT = """
Example 1:
The question is : Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?

The answer is: \\boxed{\\sqrt{5}}<end_of_reasoning>


Example 2:
The question is : Convert $10101_3$ to a base 10 integer.

The answer is: \\boxed{91}<end_of_reasoning>


Example 3:
The question is :
The points $(x, y)$ represented in this table lie on a straight line. The point $(28, t)$ lies on the same line. What is the value of $t?$ \\begin{tabular}{c|c}\n$x$ & $y$ \\\\ \\hline\n1 & 7 \\\\\n3 & 13 \\\\\n5 & 19 \\\\\n\\end{tabular}

The answer is: \\boxed{88}<end_of_reasoning>


Example 4:
The question is :
Five socks, colored blue, brown, black, red, and purple are in a drawer. In how many different ways can we choose three socks from the drawer if the order of the socks does not matter?

The answer is: \\boxed{10}<end_of_reasoning>

""".strip()

MATH_COT_4_SHOT = """
Example 1:
The question is : Gracie and Joe are choosing numbers on the complex plane. Joe chooses the point $1+2i$. Gracie chooses $-1+i$. How far apart are Gracie and Joe's points?
The reasoning steps are:

The distance between two points $(x_1,y_1)$ and $(x_2,y_2)$ in the complex plane is given by the formula $\\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$.
In this case, Joe's point is $(1,2)$ and Gracie's point is $(-1,1)$.
So the distance between their points is $\\sqrt{((-1)-(1))^2+((1)-(2))^2}=\\sqrt{(-2)^2+(-1)^2}=\\sqrt{4+1}=\\sqrt{5}$.
Therefore, Gracie and Joe's points are $\\boxed{\\sqrt{5}}$ units apart.
The answer is: \\boxed{\\sqrt{5}}<end_of_reasoning>


Example 2:
The question is : Convert $10101_3$ to a base 10 integer.
The reasoning steps are:

$10101_3 = 1 \\cdot 3^4 + 0 \\cdot 3^3 + 1 \\cdot 3^2 + 0 \\cdot 3^1 + 1 \\cdot 3^0 = 81 + 9 + 1 = \\boxed{91}$.
The answer is: \\boxed{91}<end_of_reasoning>


Example 3:
The question is :
The points $(x, y)$ represented in this table lie on a straight line. The point $(28, t)$ lies on the same line. What is the value of $t?$ \\begin{tabular}{c|c}\n$x$ & $y$ \\\\ \\hline\n1 & 7 \\\\\n3 & 13 \\\\\n5 & 19 \\\\\n\\end{tabular}

The reasoning steps are:
The slope of a line passing through two points $(x_1, y_1)$ and $(x_2, y_2)$ is given by $\\frac{y_2 - y_1}{x_2 - x_1}$.
Using the points $(1, 7)$ and $(5, 19)$ from the table, we find that the slope of the line passing through these points is $\\frac{19 - 7}{5 - 1} = \\frac{12}{4} = 3$.
Since the point $(28, t)$ lies on the same line, the slope of the line passing through $(28, t)$ and $(5, 19)$ is also $3$.
Using the slope-intercept form of a line, $y = mx + b$, where $m$ is the slope and $b$ is the $y$-intercept, we can find the equation of the line passing through $(5, 19)$ with a slope of $3$.
Substituting the coordinates of the point $(5, 19)$ into the equation, we have $19 = 3(5) + b$, which gives us $b = 19 - 15 = 4$.
Therefore, the equation of the line passing through these two points is $y = 3x + 4$.
Substituting $x = 28$ into this equation, we can find the value of $t$:
$t = 3(28) + 4 = 84 + 4 = \\boxed{88}$.
The answer is: \\boxed{88}<end_of_reasoning>


Example 4:
The question is :
Five socks, colored blue, brown, black, red, and purple are in a drawer. In how many different ways can we choose three socks from the drawer if the order of the socks does not matter?

The reasoning steps are:
This is a combination problem, since the order of the socks does not matter.
We want to choose 3 out of the 5 socks, so we can use the formula for combinations:
$\\binom{n}{k}=\\dfrac{n!}{k!(n-k)!}$.
In this case, $n=5$ (the total number of socks) and $k=3$ (the number of socks to choose).
Plugging in the values, we get $\\binom{5}{3}=\\dfrac{5!}{3!(5-3)!}=\\dfrac{5!}{3!2!}=\\dfrac{5\\times4\\times3\\times2\\times1}{3\\times2\\times1\\times2\\times1}=\\dfrac{5\\times4}{2\\times1}=\\boxed{10}$.
The answer is: \\boxed{10}<end_of_reasoning>

""".strip()