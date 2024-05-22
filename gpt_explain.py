import openai

openai.api_key = "KEY_HERE"
ls = 90
rs = 80
p = 'back not straight'
completion = openai.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a fitness expert. You should talk to me as if you are my in-person instructor. Respond with only three sentences."},
        {"role": "user", "content": "Next I'll input some angles of some joints of me doing squats, and the problem i identified with my actions."},
        {"role": "assistant", "content": "Sure, please feel free to do that. I can tell you how you should adjust your actions."},
        {"role": "user", "content": f"the angle of the left shoulder is {ls}, the angle of the right shoulder is {rs}, and i think my problem is {p}"}
    ]
)

print(completion.choices[0].message.content)