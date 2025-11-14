# Proof Tool

A real time FUD destruction engine that stress tests claims through rigorous logical analysis and evidence gathering. Not a fact checker, a proof checker. Feed it any claim and get a transparent proof tree showing logical derivation or exposing shaky foundations. Built on falsification first skepticism: it tries to break claims before defending them, shows its work, cites sources, and admits when it doesn't know.

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your OPENROUTER_API_KEY
   ```

3. **Run the agent**:
   ```bash
   python -m chat.agent
   ```
4. **Chat with the agent and ask it about anything you want proof of**

## The Story:

Frustrated with social media. Every day feeds full of confident claims, some true, some false, most somewhere in between. Grok does quick searches (can be poisoned). Community Notes are gamed. What if we could go deeper? Build something that doesn't just cite sources but actually derives truth from first principles?

The vision: real time FUD destruction engine. Ask any claim, get a proof tree in 5 seconds showing logical derivation or exposing the foundation as sand. Not a fact checker. A proof checker.

Starting simple. Just a CLI that takes text input, outputs proof tree. No UI, no polish. Core question: can AI construct rigorous proofs?

Using Grok. What better model than the truth seeker itself? Made the master prompt. This is 80% of the product. Need structured JSON: claim, assumptions, derivation steps, falsifiable tests, verdict.

First runs are weird. Model outputs fine but feels like it's justifying claims, not evaluating them. Building proofs for the claim, not stress testing them. Problem...

Made a few changes but still noticing bias. Model wants to prove claims, not test them. Looking at the prompt language: "Your purpose is to construct rigorous, verifiable proof trees for claims."

That word construct is the problem. Implies building something for the claim, not against it. Model acting like a defense attorney. Should be a skeptical scientist.

Rewriting the whole prompt. Instead of construct proofs, it's stress test claims. Instead of seeking supporting evidence first, seek falsification. Core principle: falsification first.

New prompt tells the model: Before seeking supporting evidence, identify what would disprove the claim. Your goal is to find flaws, not justify assertions.

Gonna remove confidence scoring entirely. The verdicts themselves are enough: PROVEN, DISPROVEN, UNSUPPORTED, UNVERIFIABLE. These say something about the claim, not about how the model "feels" about it.

Made test_claims.py with some sample claims. Some true, some false, some unsubstantiated. But static tests are annoying with LLMs. Same claim gives different reasoning each time. Can't just check if verdict matches expected.

Had an idea. What if I use another AI to grade the analysis? Built test_ai_grading.py. Instead of checking if verdicts match, it checks if the reasoning makes sense. Does the evidence actually support the verdict? Is the logic coherent? Are the assumptions reasonable?

This is way better. Flexible validation that can handle the non deterministic nature of LLMs.

The verdict categories kept changing. Started with vague stuff like DERIVED FROM FIRST PRINCIPLES vs MATHEMATICALLY UNSOUND. Too ambiguous.

Refined it to four clear categories with a simple decision tree. Can it be tested? No means UNVERIFIABLE. Does evidence prove it? Yes means PROVEN. Does evidence disprove it? Yes means DISPROVEN. Otherwise UNSUPPORTED.

Much clearer now. The model has explicit criteria for each verdict. Less wiggle room.

Been thinking about what this all means. The philosophy that's emerging: skepticism over justification. Find flaws, don't defend claims. Evidence over fake confidence. Show your work. Falsification over verification. Try to break things. Tools over training data. Real evidence matters. Transparency. Show every step.

It's not perfect. Not omniscient. But it's honest. Shows its work. Cites sources. Admits when it doesn't know.

Still just a proof of concept. The real vision is bigger. Imagine X with Prove button on every post. Background system building proof trees for viral claims. Make truth seeking as easy as sharing misinformation.

Considering the next iteration to make this a tool (like agent as a tool) to seperate concerns from a chat model.

After testing this a bit more... I need the date and time to be injected into the context. 

Okay its an agent as a tool now. I'm noticing that this may be able to prove alot more than just claims... It may be able to test theories, hypotheses, and even theories about theories... Gonna need help math and physics experts on that. But I wrote a few simple tests to see ifs possible... and it seems to be. So we'll see.

I want to create a UI for the outputs somehow. 