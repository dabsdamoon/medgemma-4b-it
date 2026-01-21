from transformers import pipeline
from PIL import Image
import requests
import torch

# Use float16 (native hardware support on RTX 2080, faster than bfloat16)
pipe = pipeline(
    "image-text-to-text",
    model="google/medgemma-1.5-4b-it",
    torch_dtype=torch.bfloat16,
    device="cuda",
)

# Use real X-ray image from HuggingFace example
# image_url = "https://upload.wikimedia.org/wikipedia/commons/c/c8/Chest_Xray_PA_3-8-2010.png"
# image = Image.open(requests.get(image_url, headers={"User-Agent": "example"}, stream=True).raw)

# sample_document = """
# Patient: John Doe, 65-year-old male
# Date of Visit: January 15, 2026
# Chief Complaint: Chest pain and shortness of breath

# History of Present Illness:
# The patient presents with intermittent chest pain for the past 3 days,
# described as pressure-like sensation in the substernal area, radiating
# to the left arm. Pain is exacerbated by exertion and relieved by rest.
# Associated symptoms include dyspnea on exertion and mild diaphoresis.

# Past Medical History:
# - Hypertension (diagnosed 2018)
# - Type 2 Diabetes Mellitus (diagnosed 2020)
# - Hyperlipidemia

# Current Medications:
# - Lisinopril 10mg daily
# - Metformin 500mg twice daily
# - Atorvastatin 20mg daily

# Physical Examination:
# - BP: 145/92 mmHg
# - HR: 88 bpm, regular
# - RR: 18/min
# - SpO2: 96% on room air
# - Cardiac: Regular rate and rhythm, no murmurs
# - Lungs: Clear to auscultation bilaterally

# Diagnostic Tests:
# - ECG: Sinus rhythm, ST depression in leads V4-V6
# - Troponin I: 0.08 ng/mL (elevated)
# - BNP: 180 pg/mL

# Assessment:
# Acute coronary syndrome - Non-ST elevation myocardial infarction (NSTEMI)

# Plan:
# 1. Admit to cardiac care unit
# 2. Start dual antiplatelet therapy (Aspirin + Clopidogrel)
# 3. Initiate heparin infusion
# 4. Cardiology consult for potential coronary angiography
# 5. Continue current medications
# 6. Serial troponin levels every 6 hours
# """


sample_document="""
# **Nutrition During Pregnancy**

Eating well is one of the best things you can do during pregnancy. Good nutrition
helps you handle the extra demands on your body as your pregnancy progresses.
The goal is to balance getting enough nutrients to support the growth of your _**fetus**_
and maintaining a healthy weight.


You Need to Know
### • the key vitamins and minerals • how to plan healthy meals • the five food groups • how much weight to gain during pregnancy


## What Healthy Eating Means

The popular saying is that when you’re
pregnant you should “eat for two,” but now
we know that it’s dangerous to eat twice
your usual amount of food during
pregnancy. Instead of “eating for two,”
think of it as eating twice as healthy.

**How many calories should I be**
**taking in?**
If you are pregnant with one fetus, you will
need a little more than 300 extra _**calories**_
per day starting in the second trimester
(and a bit more in the third trimester).
That’s roughly the calorie count of a glass
of skim milk and half a sandwich.

If you are pregnant with twins, you
should get about 600 extra calories a day. If
you are pregnant with triplets, you should
get 900 extra calories a day.
## Key Vitamins and Minerals During Pregnancy

Vitamins and minerals play important roles
in all of your body functions. Eating healthy
foods and taking a prenatal vitamin every
day should supply all the vitamins and
minerals you need during pregnancy.

**How many prenatal vitamins**
**should I take each day?**
Take only one serving of your prenatal
supplement each day. Read the bottle to see



how many pills make up one daily serving.
If your _**obstetrician–gynecologist (ob-gyn)**_
thinks you need an extra amount of a
vitamin or mineral, your ob-gyn may
recommend it as a separate supplement.

**Can I take more prenatal vitamins**
**to make up for a deficiency?**
No, do not take more than the recommended amount of your prenatal vitamin
per day. Some multivitamin ingredients,
such as vitamin A, can cause _**birth defects**_
at higher doses.

**What vitamins and minerals do I**
**need during pregnancy?**
During pregnancy you need _**folic acid**_,
iron, calcium, vitamin D, choline, omega-3
fatty acids, B vitamins, vitamin C, and
many other nutrients (Table 1).

**What is folic acid?**
Folic acid, also known as folate, is a B
vitamin that is important during pregnancy. Folic acid may help prevent major
birth defects of the fetus’s brain and spine
called _**neural tube defects (NTDs).**_

**How much folic acid should I take?**
When you are pregnant you need 600
micrograms (mcg) of folic acid each day.
Since it’s hard to get this much folic acid
from food alone, you should take a daily
prenatal vitamin with at least 400 mcg
starting at least 1 month before pregnancy
and during the first 12 weeks of pregnancy.



If you have already had a child with an
NTD, you should take 4 milligrams (mg) of
folic acid each day as a separate supplement
at least 3 months before pregnancy and for
the first 3 months of pregnancy. You and
your ob-gyn can discuss whether you need to
supplement with more than 400 mcg daily.

**Why is iron important during**
**pregnancy?**
Iron is used by your body to make the extra
blood that you and your fetus need during
pregnancy. When you are not pregnant, you
need 18 mg of iron per day. When you are
pregnant, you need 27 mg per day. You can
get this amount in most prenatal vitamins.

**How can I make sure I’m getting**
**enough iron?**
In addition to taking a prenatal vitamin
with iron, you should eat iron-rich foods
like beans, lentils, enriched breakfast
cereals, beef, turkey, liver, and shrimp. You
should also eat foods that help your body
absorb iron, including oranges, grapefruit,
strawberries, broccoli, and peppers.

Your blood should be tested during
pregnancy to check for _**anemia**_ . If you have
anemia, your ob-gyn may recommend extra
iron supplements.

**What is calcium?**
Calcium is a mineral that builds your fetus’s
bones and teeth. If you are 18 or younger,
you need 1,300 mg of calcium per day. If you
are 19 or older, you need 1,000 mg per day.


**What are omega-3 fatty acids?**
Omega-3 fatty acids are a type of fat found
naturally in many kinds of fish. Omega-3s
may be important for your fetus’s brain
development before and after birth.

**How much fish should I eat to get**
**the omega-3 fatty acids that I need?**
You should eat two to three servings of fish
or shellfish per week before getting pregnant,
while pregnant, and while breastfeeding. A
serving of fish is 4 ounces (oz).

**Which types of fish should I avoid?**
Some types of fish have higher levels of
mercury than others. Mercury is a metal
that has been linked to birth defects. Do not
eat bigeye tuna, king mackerel, marlin,
orange roughy, shark, swordfish, or tilefish.
Limit white (albacore) tuna to only 6 oz a
week. You should also check advisories
about fish caught in local waters.



**What foods contain calcium?**
Milk and other dairy products, such as cheese
and yogurt, are the best sources of calcium. If
you have trouble digesting milk products, you
can get calcium from other sources, such as
broccoli, fortified foods (cereals, breads, and
juices), almonds and sesame seeds, sardines
or anchovies with the bones, and dark green
leafy vegetables. You can also get calcium
from calcium supplements.

**What is vitamin D?**
Vitamin D works with calcium to help the
fetus’s bones and teeth develop. Vitamin D
is also essential for healthy skin and
eyesight. Whether you are pregnant or not,
you need 15 mcg (600 international units

[IU]) of vitamin D a day.

**What foods contain vitamin D?**
Good sources of vitamin D include
fortified milk and breakfast cereal, fatty
fish (salmon and mackerel), fish liver oils,
and egg yolks.



**How will I know I’m getting enough**
**vitamin D?**
Many people do not get enough vitamin D. If
your ob-gyn thinks you may have low levels
of vitamin D, a test can be done to check the
level in your blood. If it is below normal, you
may need to take a vitamin D supplement.

**What is choline?**
Choline plays a role in your fetus’s brain
development. It may also help prevent
some common birth defects. Experts
recommend that you get 450 mg of choline
each day during pregnancy.

**What foods contain choline?**
Choline can be found in chicken, beef, eggs,
milk, soy products, and peanuts. Although
the body produces some choline on its own,
it doesn’t make enough to meet all your
needs while you are pregnant. It’s important
to get choline from your diet because it is
not found in most prenatal vitamins.


## Table 1: Key Vitamins and Minerals During Pregnancy
































|Nutrient<br>(Daily Recommended Amount)|Why You and Your Fetus Need It|Best Sources|
|---|---|---|
|Calcium<br>(1,300 milligrams [mg] for ages 14 to 18<br>1,000 mg for ages 19 to 50)|Builds strong bones and teeth|Milk, cheese, yogurt, sardines, dark<br>green leafy vegetables|
|Iron (27 mg)|Helps red blood cells deliver<br>**_oxygen_** to your fetus|Lean red meat, poultry, fsh, dried<br>beans and peas, iron-fortifed cereals,<br>prune juice|
|Iodine (220 micrograms [mcg])|Essential for healthy brain development|Iodized table salt, dairy products,<br>seafood, meat, some breads, eggs|
|Choline (450 mg)|Important for development of<br>your fetus’s brain and spinal cord|Milk, beef liver, eggs,<br>peanuts, soy products|
|Vitamin A<br>(750 mcg for ages 14 to 18<br>770 mcg for ages 19 to 50)|Forms healthy skin and eyesight<br>Helps with bone growth|Carrots, green leafy vegetables,<br>sweet potatoes|
|Vitamin C<br>(80 mg for ages 14 to 18<br>85 mg for ages 19 to 50)|Promotes healthy gums, teeth, and bones|Citrus fruit, broccoli, tomatoes,<br>strawberries|
|Vitamin D<br>600 international units [IU] or 15 mcg|Builds your fetus’s bones and teeth<br>Helps promote healthy eyesight and skin|Sunlight, fortifed milk, fatty fsh<br>such as salmon and sardines|
|Vitamin B6 (1.9 mg)|Helps form red blood cells<br>Helps body use protein, fat, and<br>carbohydrates|Beef, liver, pork, ham, whole-grain<br>cereals, bananas|
|Vitamin B12 (2.6 mcg)|Maintains nervous system<br>Helps form red blood cells|Meat, fsh, poultry, milk (vegetarians<br>should take a supplement)|
|Folic acid (600 mcg)|Helps prevent birth defects of the brain<br>and spine<br>Supports the general growth and<br>development of the fetus and placenta|Fortifed cereal, enriched bread and<br>pasta, peanuts, dark green leafy<br>vegetables, orange juice, beans. Also,<br>take a daily prenatal vitamin with 400<br>mcg of folic acid.|



[Nutrition During Pregnancy | acog.org/WomensHealth](https://www.acog.org/womens-health?utm_source=vanity&utm_medium=web&utm_campaign=forpatients) Page 2


**What other foods contain omega-3**
**fatty acids?**
Flaxseed (ground or as oil) is a good source
of omega-3s. Other sources of omega-3s
include broccoli, cantaloupe, kidney beans,
spinach, cauliflower, and walnuts.

There are also supplements with
omega-3s, but you should talk with your
ob-gyn before taking one. High doses may
have harmful effects.

**What are B vitamins?**
B vitamins, including B1, B2, B6, B9, and
B12, are key nutrients during pregnancy.
These vitamins

- give you energy

- supply energy for your fetus’s

development

- promote good vision

- help build the _**placenta**_

**How can I get enough B vitamins?**
Your prenatal vitamin should have the
right amount of B vitamins that you need
each day. Eating foods high in B vitamins
is a good idea too, including liver, pork,
chicken, bananas, beans, and whole-grain
cereals and breads.

**What is vitamin C?**
Vitamin C is important for a healthy
immune system. It also helps build strong
bones and muscles. During pregnancy, you
should get at least 85 mg of vitamin C each
day if you are older than 19, and 80 mg if
you are younger than 19.

**What foods contain vitamin C?**
You can get the right amount of vitamin C
in your daily prenatal vitamin, and from
citrus fruits and juices, strawberries,
broccoli, and tomatoes.

**How can I get enough water during**
**pregnancy?**
Drink throughout the day, not just when you
are thirsty. Aim for 8 to 12 cups of water a
day during pregnancy.
## How to Plan Healthy Meals During Pregnancy

There are many tools that can help you plan
healthy meals. One useful tool is the MyPlate
food-planning guide from the U.S. Department of Agriculture. The MyPlate website,
[www.myplate.gov, can help you learn how to](https://www.myplate.gov/)
make healthy food choices at every meal.



**How can MyPlate help me plan**
**healthy meals?**
The MyPlate website, [www.myplate.gov,](https://www.myplate.gov/)
offers a MyPlate Plan, which shows how
much to eat based on how many calories
you need each day. The MyPlate Plan is
personalized based on your

- height

- prepregnancy weight

- physical activity level

The MyPlate Plan can help you learn about
choosing foods from each food group to
get the vitamins and minerals you need
during pregnancy. The MyPlate Plan can
also help you limit calories from added
sugars and saturated fats.
## The Five Food Groups

**What are the five food groups?**


- Grains

- Fruits

- Vegetables

- Protein foods

- Dairy foods

**What are grains?**
Bread, pasta, oatmeal, cereal, and tortillas
are all grains. Whole grains are those that
haven’t been processed and include the
whole grain kernel. Oats, barley, quinoa,
brown rice, and bulgur are all whole grains,
as are products made with those grains.

Look for the words “whole grain” on the
product label. When you plan meals, make
half of your grain servings whole grains.

**What types of fruit should I eat?**
You can eat fresh, canned, frozen, or dried
fruit. Juice that is 100 percent fruit juice
counts in the fruit category, but it is best to
eat mostly whole fruit instead of juice.

## Resources



**MyPlate**
Healthy eating resources from the U.S. Department of Agriculture (USDA).
[www.myplate.gov](https://www.myplate.gov/)

[• Healthy Eating on a Budget: www.myplate.gov/eat-healthy/healthy-eating-budget](https://www.myplate.gov/eat-healthy/healthy-eating-budget)

[• Pregnancy and Breastfeeding: www.myplate.gov/life-stages/pregnancy-and-](https://www.myplate.gov/life-stages/pregnancy-and-breastfeeding)

[breastfeeding](https://www.myplate.gov/life-stages/pregnancy-and-breastfeeding)

- MyPlate Plan: [www.myplate.gov/myplate-plan](https://www.myplate.gov/myplate-plan)

**Food Sources of Select Nutrients**
Examples of foods that are good sources of important nutrients.
[www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials/](https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials/food-sources-select-nutrients)
[food-sources-select-nutrients](https://www.dietaryguidelines.gov/resources/2020-2025-dietary-guidelines-online-materials/food-sources-select-nutrients)



Make half your plate fruit and vegetables
during mealtimes.

**What types of vegetables should**
**I eat?**
You can eat raw, canned, frozen, or dried
vegetables or 100 percent vegetable juice.
Use dark leafy greens to make salads.
Make half your plate fruit and vegetables
during mealtimes.

**What are protein foods?**
Meat, poultry, seafood, beans, peas, eggs,
processed soy products, nuts, and seeds all
contain protein. Eat a variety of proteins
each day.

**What are dairy foods?**
Milk and milk products, such as cheese
and yogurt, make up the dairy group.
Make sure any dairy foods you eat are
pasteurized. Choose fat-free or low-fat
(1 percent) varieties.

**Why are oils and fats important?**
Oils and fats are another part of healthy
eating. Although they are not a food group,
they do give you important nutrients.
During pregnancy, the fats that you eat
provide energy and help build the placenta
and many fetal organs.

**What are healthy sources of oils**
**and fats?**
Oils in food come mainly from plant
sources, such as olive oil, nut oils, and
grapeseed oil. They can also be found in
certain foods, such as some fish, avocados,
nuts, and olives.

Most of the fats and oils in your diet
should come from plant sources. Limit
solid fats, such as those from animal
sources. Solid fats can also be found in
processed foods.



[Nutrition During Pregnancy | acog.org/WomensHealth](https://www.acog.org/womens-health?utm_source=vanity&utm_medium=web&utm_campaign=forpatients) Page 3


## Table 2: Weight Gain During Pregnancy











|Body Mass Index (BMI)<br>Before Pregnancy|Rate of Weight Gain<br>in the Second and Third<br>Trimesters*<br>(Pounds Per Week)|Recommended Total Weight<br>Gain With a Single Fetus<br>(in Pounds)|Recommended Weight Gain<br>With Twins (in Pounds)|
|---|---|---|---|
|Less than 18.5 (underweight)|1.0 to 1.3|28 to 40|Not known|
|18.5 to 24.9<br>(normal weight)|0.8 to 1.0|25 to 35|37 to 54|
|25.0 to 29.9(overweight)|0.5 to 0.7|15 to 25|31 to 50|
|30.0 and above (obese)|0.4 to 0.6|11 to 20|25 to 42|


*Assumes a first-trimester weight gain between 1.1 and 4.4 pounds
Source: Institute of Medicine and National Research Council. 2009. Weight Gain During Pregnancy: Reexamining the Guidelines.
Washington, DC: The National Academies Press.


## Weight Gain During Pregnancy

Weight gain depends on your health and
your _**body mass index (BMI)**_ before you
were pregnant. If you were underweight
before pregnancy, you should gain more
weight than if you had a normal weight
before pregnancy. If you were overweight or
obese before pregnancy, you should gain less
weight. The amount of weight gain differs by
_**trimester**_ . Read Table 2 for recommended
weight gain during pregnancy.

**How much weight should I gain**
**during the first trimester?**
During your first 12 weeks of pregnancy—
the first trimester—you might gain only
1 to 5 pounds or none at all.

**How much should I gain during**
**the second and third trimesters?**
If you were a healthy weight before
pregnancy, you should gain a half-pound
to 1 pound per week in your second and
third trimesters.

**How many extra calories should**
**I eat?**
During the first trimester with one fetus,
usually no extra calories are needed. In the
second trimester, you will need an extra
340 calories per day, and in the third
trimester, about 450 extra calories a day.

**How can I get those extra calories?**
To get the extra calories during the day,
have healthy snacks on hand, such as nuts,
yogurt, and fresh fruit.



**What if I am obese or overweight?**
You and your ob-gyn should work together
to develop a nutrition and exercise plan. If
you are gaining less than what the guidelines suggest, and if your fetus is growing
well, gaining less than the recommended
guidelines can have benefits. If your fetus is
not growing well, changes may need to be
made to your diet and exercise plan.
## Your Takeaways

1. Eating well during your pregnancy is

one of the best things you can do for
yourself and your fetus.

2. You need to balance getting enough

nutrients to fuel the fetus’s growth with
maintaining a healthy pregnancy weight.

3. A balanced diet includes key vitamins

and minerals plus 8 to 12 cups of water
a day.



_**Neural Tube Defects (NTDs):**_ Birth defects
that result from a problem in development of
the brain, spinal cord, or their coverings.



**How can being overweight or obese**
**cause problems during pregnancy?**
Excess weight during pregnancy is associated with several pregnancy and childbirth
_**complications**_, including



_**Anemia:**_ Abnormally low levels of red blood
cells in the bloodstream. Most cases are
caused by iron deficiency (lack of iron).


## Terms You Should Know




- _**high blood pressure**_



_**Birth Defects:**_ Physical problems that are
present at birth.




- _**preeclampsia**_

- _**preterm**_ birth




- _**gestational diabetes**_



Obesity during pregnancy also increases
the risk of



_**Body Mass Index (BMI):**_ A number calculated from height and weight. BMI is used to
determine whether a person is underweight,
normal weight, overweight, or obese.




- a larger than normal fetus _**(macrosomia)**_



_**Calories:**_ Units of heat used to express the
fuel or energy value of food.




- birth injury




- _**cesarean birth**_




- birth defects, especially NTDs



_**Cesarean Birth:**_ Birth of a fetus from the
uterus through an incision (cut) made in the
woman’s abdomen.

_**Complications:**_ Diseases or conditions that
happen as a result of another disease or condition. An example is pneumonia that occurs
as a result of the flu. A complication also can
occur as a result of a condition, such as pregnancy. An example of a pregnancy complication is preterm labor.



_**Fetus:**_ The stage of human development
beyond 8 completed weeks after fertilization.



_**Folic Acid:**_ A vitamin that reduces the risk of
certain birth defects when taken before and
during pregnancy.



_**Gestational Diabetes:**_ Diabetes that starts
during pregnancy.



_**High Blood Pressure:**_ Blood pressure above
the normal level. Also called hypertension.



_**Macrosomia:**_ A condition in which a fetus
grows more than expected, often weighing
more than 8 pounds and 13 ounces (4,000
grams).



[Nutrition During Pregnancy | acog.org/WomensHealth](https://www.acog.org/womens-health?utm_source=vanity&utm_medium=web&utm_campaign=forpatients) Page 4


_**Preterm:**_ Less than 37 weeks of pregnancy.

_**Trimester:**_ A 3-month time in pregnancy. It
can be first, second, or third.



_**Obstetrician–Gynecologist (Ob-Gyn):**_ A
doctor with special training and education in
women’s health.



_**Oxygen:**_ An element that we breathe in to
sustain life.



_**Placenta:**_ An organ that provides nutrients to
and takes waste away from the fetus.



_**Preeclampsia:**_ A disorder that can occur during
pregnancy or after childbirth in which there is
high blood pressure and other signs of organ
injury. These signs include an abnormal amount
of protein in the urine, a low number of platelets, abnormal kidney or liver function, pain
over the upper abdomen, fluid in the lungs, or a
severe headache or changes in vision.



This information is designed as an educational aid for the public. It offers current information and opinions related to women’s health. It is not intended
as a statement of the standard of care. It does not explain all of the proper treatments or methods of care. It is not a substitute for the advice of a
[physician. For ACOG’s complete disclaimer, visit www.acog.org/WomensHealth-Disclaimer.](http://www.acog.org/WomensHealth-Disclaimer)

Copyright June 2023 by the American College of Obstetricians and Gynecologists. All rights reserved. No part of this publication may be reproduced, stored
in a retrieval system, posted on the internet, or transmitted, in any form or by any means, electronic, mechanical, photocopying, recording, or otherwise,
without prior written permission from the publisher.

This is EP001 in ACOG’s Patient Education Pamphlet Series.

ISSN 1074-8601

American College of Obstetricians and Gynecologists
409 12th Street SW
Washington, DC 20024-2188


[Nutrition During Pregnancy | acog.org/WomensHealth](https://www.acog.org/womens-health?utm_source=vanity&utm_medium=web&utm_campaign=forpatients) Page 5
"""

prompt = """
You are an expert medical professional.
Provide a clear, accurate, and concise summary of the following medical document.
Focus on key findings, diagnoses, treatments, and recommendations. Summarize medical-related contents only.\n\n
f"Medical Document:\n{text}\n\n
"""


messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt.format(text=sample_document)},
        ]
    }
]

# messages = [
#     {
#         "role": "user",
#         "content": [
#             {"type": "image", "image": image},
#             {"type": "text", "text": "Describe this X-ray"}
#         ]
#     }
# ]

output = pipe(text=messages, max_new_tokens=2000, do_sample=False)
print(output[0]["generated_text"][-1]["content"])
