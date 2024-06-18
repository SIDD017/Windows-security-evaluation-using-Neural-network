from pr2_sj6670 import SecurityEvaluation
import random

untrusted_apps = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
unsigned_drivers = [0, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25]
logoff = [0, 1]
pasword_length = [0, 4, 8, 12, 16]
password_duration = [0, 45, 90, 135, 180]
intact_firewall = [0, 1]
antivirus = [0, 1]
updates = [0, 1, 2, 3, 4, 5]
encryption = [0, 1]
dataset_facts = []
dataset_scores = []

for i in range(0, 50):
    f_logoff = random.choice(logoff)
    f_firewall = random.choice(intact_firewall)
    f_antivirus = random.choice(antivirus)
    f_encryption = random.choice(encryption)
    w_logoff = 0
    w_firewall = 0
    w_antivirus = 0
    w_encryption = 0


    if f_logoff == 0:
        w_logoff = False
    else:
        w_logoff = True

    if f_firewall == 0:
        w_firewall = False
    else:
        w_firewall = True

    if f_antivirus == 0:
        w_antivirus = False
    else:
        w_antivirus = True

    if f_encryption == 0:
        w_encryption = False
    else:
        w_encryption = True
    
    facts = {
        "untrusted_apps": random.choice(untrusted_apps),
        "unsigned_drivers": random.choice(unsigned_drivers),
        "pasword_length": random.choice(pasword_length),
        "password_duration": random.choice(password_duration),
        "logoff": w_logoff,
        "intact_firewall": w_firewall,
        "antivirus": w_antivirus,
        "updates": random.choice(updates),
        "encryption": w_encryption
    }

    dataset_facts.append([facts["untrusted_apps"], 
                          facts["unsigned_drivers"],
                          facts["pasword_length"],
                          facts["password_duration"],
                          f_logoff,
                          f_firewall,
                          f_antivirus,
                          facts["updates"],
                          f_encryption]) 

    engine = SecurityEvaluation(facts)
    engine.reset()
    engine.run()
    dataset_scores.append([engine.finalscore])

print(dataset_facts)
print(dataset_scores)