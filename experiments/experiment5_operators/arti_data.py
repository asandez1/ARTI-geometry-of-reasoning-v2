#!/usr/bin/env python3
"""
Data Collection and Labeling Pipeline for ARTI Training (Scaled v2).

Generates labeled reasoning traces from three sources:
1. Built-in examples from reasoning_types.py (24 pairs, 8 types x 3 each)
2. Expanded synthetic traces: 50-80 diverse filled-in sentences per type
3. Benchmark-derived traces from GSM8K, ARC, StrategyQA, FOLIO with
   heuristic labeling + class-balanced sampling

v2 Changes (scaling to 5K-10K):
- Replaced placeholder templates with 50+ real sentences per type
- Added class-balanced sampling (cap overrepresented, oversample underrepresented)
- Increased benchmark pull limits (200 -> full train splits)
- Added class-weighted loss support
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np
import logging
import json
from pathlib import Path
from collections import Counter

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.reasoning_types import (
    ReasoningType, N_REASONING_TYPES, REASONING_TYPES,
    HeuristicLabeler, get_all_examples,
)

logger = logging.getLogger(__name__)


# ─── Reasoning Trace Dataset ──────────────────────────────────────────────────

class ReasoningTraceDataset(Dataset):
    """
    Dataset of reasoning traces with type labels.

    Each sample has:
    - embedding: [encoder_dim] sentence-transformer embedding of the text
    - label: int (ReasoningType enum value)
    - soft_label: [N_REASONING_TYPES] probability distribution
    - text: str (original text, for debugging)
    - source: str (where the trace came from)
    """

    def __init__(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        soft_labels: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
    ):
        self.embeddings = embeddings
        self.labels = labels
        self.soft_labels = soft_labels
        self.texts = texts or [""] * len(labels)
        self.sources = sources or ["unknown"] * len(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            'embeddings': self.embeddings[idx],
            'labels': self.labels[idx],
        }
        if self.soft_labels is not None:
            item['soft_labels'] = self.soft_labels[idx]
        return item

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        labels: List[int],
        encoder,
        soft_labels: Optional[List[Dict]] = None,
        sources: Optional[List[str]] = None,
        batch_size: int = 64,
    ) -> 'ReasoningTraceDataset':
        """Create dataset by encoding texts with a sentence-transformer."""
        with torch.no_grad():
            embeddings = encoder.encode_texts(texts, batch_size=batch_size)

        labels_tensor = torch.tensor(labels, dtype=torch.long)

        soft_tensor = None
        if soft_labels is not None:
            soft_tensor = torch.zeros(len(labels), N_REASONING_TYPES)
            for i, sl in enumerate(soft_labels):
                for rtype, prob in sl.items():
                    soft_tensor[i, int(rtype)] = prob

        return cls(
            embeddings=embeddings,
            labels=labels_tensor,
            soft_labels=soft_tensor,
            texts=texts,
            sources=sources,
        )

    def split(
        self, train_ratio: float = 0.8, seed: int = 42
    ) -> Tuple['ReasoningTraceDataset', 'ReasoningTraceDataset']:
        """Split into train/val sets with stratified sampling."""
        rng = np.random.RandomState(seed)
        n = len(self)
        indices = rng.permutation(n)

        train_indices = []
        val_indices = []
        for label_val in range(N_REASONING_TYPES):
            label_mask = (self.labels == label_val).numpy()
            label_indices = indices[label_mask[indices]]
            split_point = int(len(label_indices) * train_ratio)
            train_indices.extend(label_indices[:split_point].tolist())
            val_indices.extend(label_indices[split_point:].tolist())

        rng.shuffle(train_indices)
        rng.shuffle(val_indices)

        def subset(idxs):
            return ReasoningTraceDataset(
                embeddings=self.embeddings[idxs],
                labels=self.labels[idxs],
                soft_labels=self.soft_labels[idxs] if self.soft_labels is not None else None,
                texts=[self.texts[i] for i in idxs],
                sources=[self.sources[i] for i in idxs],
            )

        return subset(train_indices), subset(val_indices)


# ─── Data Sources ──────────────────────────────────────────────────────────────

def collect_builtin_examples() -> Tuple[List[str], List[int], List[str]]:
    """Collect built-in examples from reasoning_types.py (24 step-pairs)."""
    texts, labels, sources = [], [], []
    for step1, step2, rtype in get_all_examples():
        texts.append(f"{step1} {step2}")
        labels.append(int(rtype))
        sources.append("builtin")
    return texts, labels, sources


# ─── Expanded Synthetic Examples (v2) ─────────────────────────────────────────
# Real, diverse, filled-in sentences organized by reasoning type.
# Each sentence clearly exhibits its reasoning type across varied domains.

EXPANDED_EXAMPLES = {

    ReasoningType.PHYSICAL_CAUSE: [
        # Physics
        "Heating the metal rod caused it to expand by 2 millimeters due to thermal expansion.",
        "The asteroid impact generated a massive dust cloud that blocked sunlight for months.",
        "Increasing voltage across the resistor resulted in a proportional increase in current.",
        "The sudden pressure drop caused the liquid to boil at room temperature.",
        "Exposure to ultraviolet radiation damages DNA strands and leads to skin cancer.",
        "The earthquake ruptured the gas main, which led to fires across the district.",
        "The magnetic field reversal caused navigation instruments to give false readings.",
        "Heavy rainfall upstream produced flash flooding in the valley towns below.",
        "The sharp turn at high speed caused centrifugal force to push the vehicle off the road.",
        "The solar flare disrupted satellite communications across the northern hemisphere.",
        "High humidity caused condensation on the windows every morning.",
        "Mixing bleach and ammonia produces toxic chloramine gas through a chemical reaction.",
        "Friction between the tectonic plates caused the earthquake that measured 6.5 on the Richter scale.",
        "The supercooled water crystallized instantly upon contact with the ice nucleation surface.",
        "Gravity pulled the satellite into a lower orbit, increasing atmospheric drag.",
        "The laser beam heated the sample to 3000 degrees, causing it to vaporize.",
        "Sound waves reflected off the canyon walls, creating a noticeable echo effect.",
        "The electrical short circuit generated sparks that ignited the flammable solvent.",
        "Corrosion of the iron bridge supports weakened the structure until it collapsed.",
        "The magnetic field induced an electric current in the moving conductor.",
        # Chemistry / Biology
        "Ocean acidification due to rising CO2 levels is destroying coral reef ecosystems.",
        "The allergic reaction was caused by trace amounts of peanut protein in the meal.",
        "Smoking damaged the alveoli, which resulted in reduced lung capacity.",
        "The patient's chronic inflammation was caused by an autoimmune response attacking healthy tissue.",
        "Administering the vaccine triggered an immune response that prevented infection.",
        "The enzyme catalyzed the reaction, lowering the activation energy required.",
        "Excessive fertilizer runoff caused algal blooms that depleted oxygen in the lake.",
        "The virus mutated its spike protein, which allowed it to evade existing antibodies.",
        "Photosynthesis converts carbon dioxide and water into glucose using solar energy.",
        "The antibiotic disrupted bacterial cell wall synthesis, causing the bacteria to lyse.",
        "Dehydration caused the patient's blood pressure to drop dangerously low.",
        "The radiation exposure damaged bone marrow cells, leading to reduced blood cell production.",
        # Geology / Weather
        "Poor soil drainage caused root rot in the newly planted trees.",
        "Deforestation of the hillside caused severe mudslides during the rainy season.",
        "Overgrazing by livestock caused severe topsoil erosion on the hillside.",
        "The late frost destroyed the early-blooming cherry blossoms through ice crystal damage.",
        "Volcanic ash in the atmosphere scattered sunlight and caused the temperature to drop globally.",
        "The glacial melt increased river flow, flooding downstream communities.",
        "Wind erosion carved the sandstone into distinctive arch formations over millions of years.",
        "The underground aquifer dried up because excessive pumping exceeded the recharge rate.",
    ],

    ReasoningType.BEHAVIORAL_CAUSE: [
        # Economics / Business
        "The central bank raised interest rates, which caused housing prices to decline.",
        "Supply chain disruptions from poor planning led to widespread shortages and price increases.",
        "The tariff on imported steel drove up manufacturing costs for domestic automakers.",
        "Inflation eroded purchasing power, so consumer spending dropped by 4 percent.",
        "The tech bubble burst because investors overvalued companies relative to their earnings.",
        "Automation in factories resulted in higher productivity but fewer manufacturing jobs.",
        "The interest rate cut stimulated borrowing and boosted economic growth.",
        "The new policy reduced emissions by 30 percent because factories adopted cleaner technology.",
        "The CEO's decision to pivot to mobile led to a 200 percent increase in user engagement.",
        "Cutting the research budget resulted in fewer patent filings the following year.",
        "The marketing team's rebranding campaign increased brand recognition by 45 percent.",
        "Imposing sanctions on the trading partner disrupted bilateral commerce for years.",
        "The startup's aggressive pricing strategy captured 30 percent market share within a year.",
        "Subsidizing renewable energy encouraged companies to invest in solar panel manufacturing.",
        "The merger created a monopoly that raised prices for consumers across the sector.",
        # Everyday / Personal
        "Leaving the window open during the storm caused water damage to the floor.",
        "Forgetting to water the garden for two weeks killed most of the tomato plants.",
        "She practiced piano daily for a year, which resulted in winning the competition.",
        "Adding too much salt to the soup made it inedible.",
        "Running in wet shoes caused painful blisters on both feet.",
        "He forgot to set an alarm and overslept, missing his flight.",
        "The coach benched the star player, which cost the team the championship.",
        "She invested her savings in index funds and built significant wealth over 20 years.",
        "The student chose to skip classes, which led to failing the course.",
        "Volunteering at the hospital inspired her to pursue a career in medicine.",
        "The driver's decision to text while driving caused the rear-end collision.",
        "Hiring an experienced project manager reduced delivery delays by 60 percent.",
        # Politics / Social
        "The government's austerity measures led to widespread public protests.",
        "Deregulating the banking industry contributed to risky lending practices.",
        "The teacher's encouragement motivated the student to apply to top universities.",
        "Neglecting infrastructure maintenance caused the bridge to become structurally unsafe.",
        "The city's investment in public transit reduced commute times by 25 percent.",
        "Banning single-use plastics forced retailers to adopt biodegradable alternatives.",
        "The journalist's investigation exposed corruption that led to the official's resignation.",
        "Poor management decisions caused the once-profitable company to file for bankruptcy.",
        "The foundation's scholarship program enabled 500 students to attend college.",
        "The landlord's refusal to fix the heating caused tenants to file a formal complaint.",
    ],

    ReasoningType.SYSTEMIC_CAUSE: [
        # Ecology
        "Removing the keystone predator caused herbivore overpopulation, which destroyed vegetation and led to ecosystem collapse.",
        "The introduction of predators to the island triggered a cascade that led to the extinction of native birds.",
        "Deforestation disrupted the water cycle, which reduced rainfall, which further accelerated desertification in a vicious cycle.",
        "Pesticide use killed pollinating insects, which reduced crop yields, which increased pesticide use in a downward spiral.",
        "Coral bleaching weakened reef structures, which reduced fish habitats, which collapsed the local fishing industry.",
        "Invasive species outcompeted native plants, which eliminated food sources for native animals, triggering a biodiversity collapse.",
        "Ocean warming reduced oxygen levels, which killed deep-sea organisms, which disrupted the entire marine food chain.",
        # Economics / Finance
        "A single bank failure triggered a chain reaction of defaults that cascaded through the interconnected financial system.",
        "The housing market crash caused mortgage defaults, which caused bank failures, which caused a credit freeze across the economy.",
        "Rising oil prices increased transportation costs, which raised food prices, which reduced consumer spending across all sectors.",
        "The factory closure caused unemployment, which reduced local spending, which forced other businesses to close in a downward spiral.",
        "Currency devaluation increased import costs, which raised inflation, which further devalued the currency in a feedback loop.",
        "Trade war escalation triggered retaliatory tariffs that cascaded through global supply chains affecting dozens of industries.",
        "The stock market crash triggered margin calls, which forced selling, which deepened the crash in a self-reinforcing cycle.",
        # Technology / Infrastructure
        "A memory leak in the application caused the server to crash, which brought down dependent services, cascading across the platform.",
        "The buffer overflow vulnerability allowed attackers to gain access, which they used to compromise the entire network.",
        "Network congestion caused packet loss, which triggered retransmissions, which amplified the congestion further.",
        "A software update introduced a bug that corrupted the database, which caused downstream services to fail in sequence.",
        "The power grid failure cascaded across interconnected regions, leaving millions without electricity for days.",
        "A single point of failure in the DNS system propagated outages across thousands of websites.",
        "The ransomware spread laterally through the network, encrypting servers one by one until the entire system was locked.",
        # Epidemiology / Social
        "The initial outbreak spread to travelers, who carried it to new cities, creating secondary outbreaks across continents.",
        "Misinformation on social media amplified vaccine hesitancy, which increased disease spread, which generated more fear and misinformation.",
        "The refugee crisis overwhelmed border resources, which slowed processing, which created camps that became humanitarian disasters.",
        "Lack of proper ventilation in the building caused mold growth, which triggered respiratory issues, which led to a public health investigation.",
        "The drought killed crops, causing food shortages that triggered mass migration, which destabilized neighboring regions.",
        # Geopolitics
        "The assassination triggered a cascade of alliance activations that drew the entire continent into war.",
        "Economic sanctions weakened the regime, which caused internal unrest, which led to a military coup with regional consequences.",
        "Climate change reduced crop yields in vulnerable regions, which caused food insecurity, which fueled conflict and mass displacement.",
        "The dam collapse released floodwaters that destroyed bridges downstream, cutting off supply routes and isolating entire communities.",
        # Technology cascades
        "A failed component in the cooling system caused the reactor to overheat, which triggered safety shutdowns across the entire plant.",
        "The satellite failure disrupted GPS signals, which affected air traffic control, which caused delays at airports worldwide.",
        "A single typo in the configuration file propagated through the deployment pipeline and brought down production servers globally.",
        "The chipmaker's production halt caused automobile manufacturers to pause assembly lines, which created vehicle shortages lasting months.",
        "A fire at a key semiconductor factory disrupted supply chains that rippled through the electronics industry for over a year.",
        "Social media algorithms amplified extreme content, which increased polarization, which reduced trust in institutions, which further radicalized users.",
    ],

    ReasoningType.DEDUCTION: [
        # Formal logic
        "All reptiles are cold-blooded. A lizard is a reptile. Therefore, a lizard is cold-blooded.",
        "If a number is divisible by 6, it is divisible by 3. 42 is divisible by 6. Therefore, 42 is divisible by 3.",
        "Every equilateral triangle has three equal angles. This shape is equilateral. Therefore, its angles are equal.",
        "All citizens over 18 have the right to vote. Maria is 25. Therefore, Maria has the right to vote.",
        "If it snows, schools close. It is snowing. Therefore, schools are closed.",
        "No mammals lay eggs except monotremes. A dolphin is not a monotreme. Therefore, dolphins do not lay eggs.",
        "All noble gases are chemically inert. Helium is a noble gas. Therefore, helium is chemically inert.",
        "If the test is positive, the patient is infected. The test came back positive. Therefore, the patient is infected.",
        # Math
        "Given that x + 5 = 12, we can deduce that x must equal 7.",
        "Since the triangle has a right angle and sides 3 and 4, by the Pythagorean theorem the hypotenuse must be 5.",
        "If two angles of a triangle sum to 110 degrees, the third angle must be 70 degrees.",
        "All prime numbers greater than 2 are odd. 13 is prime and greater than 2. Hence, 13 is odd.",
        "From the axioms of group theory, the identity element must be unique.",
        "If the determinant of the matrix is zero, it follows that the matrix is singular.",
        "The sum of interior angles in any polygon with n sides is (n-2) times 180 degrees. A pentagon has 5 sides. Therefore, its angles sum to 540 degrees.",
        # Science
        "All acids have a pH below 7. This solution has a pH of 3. Therefore, it is acidic.",
        "Metals conduct electricity. Copper is a metal. Therefore, copper conducts electricity.",
        "If the acceleration is zero, the velocity must be constant by Newton's first law.",
        "Given that F equals ma and the mass is 10 kg with acceleration 5 m/s squared, the force must be 50 Newtons.",
        "All viruses require a host cell to replicate. SARS-CoV-2 is a virus. Therefore, it requires a host cell.",
        # Law / Rules
        "Contracts require mutual consent. This agreement lacks consent from one party. Therefore, it is not a valid contract.",
        "All employees must complete safety training before starting work. John has not completed training. Therefore, John cannot start work.",
        "Vehicles exceeding 3.5 tons are prohibited on this bridge. This truck weighs 5 tons. Therefore, it cannot cross the bridge.",
        "If a student misses more than 5 classes, they fail automatically. Alex missed 7 classes. Therefore, Alex fails.",
        # Everyday
        "All store locations close at 9 PM on weekdays. It is Tuesday at 9:30 PM. Therefore, the store is closed.",
        "If the battery is dead, the phone won't turn on. The battery is completely drained. Therefore, the phone won't turn on.",
        "Every book in this series has exactly 12 chapters. This is a book from the series. Therefore, it has 12 chapters.",
        "Since all flights from this airport today are delayed, and my flight departs from this airport today, my flight must be delayed.",
        "All organic compounds contain carbon. Glucose is an organic compound. Therefore, glucose contains carbon.",
        "If a number ends in 0 or 5, it is divisible by 5. The number 135 ends in 5. Therefore, 135 is divisible by 5.",
        # More
        "Given the premises that all dogs are animals and Fido is a dog, it follows necessarily that Fido is an animal.",
        "If the temperature drops below zero and the road is wet, then ice will form. Both conditions are met. Therefore, ice will form.",
        "Every element in group 18 has a full outer electron shell. Neon is in group 18. Therefore, neon has a full outer shell.",
        "All squares are rectangles with equal sides. This shape has four equal sides and four right angles. Therefore, it is a square.",
        "The syllogism is valid: all birds have feathers, a penguin is a bird, therefore a penguin has feathers.",
        "Since the function is continuous on a closed interval and differentiable on the open interval, by the mean value theorem there exists a point where the derivative equals the average rate of change.",
        "Given that pressure is inversely proportional to volume at constant temperature, doubling the pressure must halve the volume.",
        "If the premises are true and the argument is valid, then the conclusion must be true.",
        "Since all integers are either even or odd, and 17 is not even, 17 must be odd.",
        "Every sample in the dataset has 384 dimensions. This embedding came from the dataset. Therefore, it has 384 dimensions.",
        "If water reaches 100 degrees Celsius at sea level, it boils. The water temperature is 100 degrees. Therefore, it is boiling.",
        "All functions in this module return tensors. predict() is a function in this module. Therefore, predict() returns a tensor.",
    ],

    ReasoningType.INDUCTION: [
        # Science
        "We observed that copper, silver, and gold all conduct electricity. This pattern suggests metals generally conduct electricity.",
        "Every swan we have seen in Europe is white. From these observations, we generalize that swans in Europe are white.",
        "Experiments 1 through 50 all showed the enzyme works best at pH 7. The evidence indicates pH 7 is the optimal condition.",
        "The drug reduced symptoms in 95 out of 100 patients tested. The data suggests it is generally effective.",
        "In each of the 200 trials, the pendulum period depended only on length. This consistent pattern shows period is independent of mass.",
        "Every rock sample from this region contained quartz. We infer that quartz is abundant in this geological formation.",
        "All 30 measurements of the speed of light gave values near 3 times 10 to the 8 meters per second. The data consistently points to a universal constant.",
        "Each bacterial colony exposed to the antibiotic showed growth inhibition. This pattern suggests broad susceptibility.",
        # Everyday
        "Every morning for the past month, the bus arrived at 8:15. I expect it will arrive at 8:15 tomorrow.",
        "Each time I eat shellfish, I get a rash. Based on this pattern, I am likely allergic to shellfish.",
        "The cafe has been crowded every Saturday I visited. I generalize that Saturdays are their busiest day.",
        "Every winter, the lake freezes by December. Based on past years, it will freeze again this December.",
        "My last five packages from that store arrived in two days. I expect the next one will also arrive in two days.",
        "The pattern shows that each new iPhone is released in September. I predict the next one will launch in September too.",
        "In every game this season, the team scored at least two goals. The trend suggests they will score again next game.",
        "Every time I water the plant on Monday, it blooms by Friday. This regular pattern suggests a five-day growth cycle.",
        # Statistics / Data
        "The survey of 1000 respondents found 67 percent prefer option A. We generalize this preference to the broader population.",
        "Crime rates have declined every year for the past decade. The consistent trend suggests the decline will continue.",
        "All 50 sampled products passed quality control. We infer the production batch is generally within specification.",
        "Test scores improved after implementing the new curriculum in schools across 12 different districts. The evidence points to the curriculum being effective.",
        "The correlation between exercise and mood improvement was significant across all 8 studies reviewed. This pattern suggests a general relationship.",
        "Customer satisfaction scores have increased each quarter for two years. The trend indicates our improvements are working.",
        # Nature
        "Every migratory bird we tagged flew south in October. We generalize that this species migrates southward in autumn.",
        "In every forest plot we sampled, biodiversity increased with distance from the road. This pattern suggests roads reduce biodiversity.",
        "Each tidal observation this month showed higher tides during the full moon. The data consistently supports the lunar tidal effect.",
        "Fossil records from five continents show the same extinction boundary. The global pattern suggests a single catastrophic event.",
        # Technology
        "After testing 200 inputs, the model consistently failed on adversarial examples. The pattern indicates a systematic vulnerability.",
        "Server response times increased every time we added more users. The data shows load scales linearly with user count.",
        "Each deployment to production this quarter went smoothly. Based on this track record, the next deployment will likely succeed.",
        "Every A/B test we ran showed the shorter form had higher conversion rates. The evidence suggests brevity improves conversion.",
        # More
        "Observations from 100 different galaxies all show redshift increasing with distance. The pattern supports the expanding universe hypothesis.",
        "In each historical example, nations that devalued their currency saw short-term export growth. The evidence suggests a general mechanism.",
        "Every student who completed the practice problems scored above 80 on the exam. This pattern suggests practice correlates with performance.",
        "Soil samples from all 15 test plots showed elevated nitrogen after fertilization. The data consistently shows fertilizer increases soil nitrogen.",
        "All observed instances of this chemical reaction produced heat. We inductively conclude the reaction is exothermic.",
        "Each time interest rates dropped, housing applications increased within two months. This repeated pattern suggests a causal relationship.",
        "The algorithm performed well on English, French, and Spanish text. The pattern suggests it generalizes across Romance languages.",
        "Across all 20 participating schools, attendance improved after free lunch programs were introduced. The evidence points to a systematic effect.",
        "Every measured sample from the spring water showed mineral content above 500 ppm. We generalize that this spring is mineral-rich.",
        "Repeated experiments under different conditions all yielded the same melting point for this compound. The consistency suggests it is an intrinsic property.",
    ],

    ReasoningType.ANALOGY: [
        # Cross-domain structural mapping
        "Just as a firewall protects a computer network from intruders, a moat protects a castle from invaders.",
        "The flow of electrical current through a wire is analogous to the flow of water through a pipe.",
        "An atom's electrons orbiting the nucleus is similar to planets orbiting the sun in our solar system.",
        "A cell membrane functions like a security checkpoint, controlling what enters and exits.",
        "The human brain's neural network processes information in the same way a computer's CPU processes data.",
        "Just as a conductor leads an orchestra, a project manager coordinates the different teams.",
        "The immune system is like an army defending the body against foreign invaders.",
        "DNA serves as a blueprint for organisms, similar to how an architect's plans serve as blueprints for buildings.",
        "The stock market fluctuates like ocean waves, with periods of calm followed by turbulent surges.",
        # Science
        "Just as light bends when passing through a prism, sound bends when moving through layers of different temperatures.",
        "A catalyst in chemistry works like a matchmaker at a party, bringing reactants together without being consumed.",
        "The relationship between voltage and current mirrors the relationship between pressure and flow rate in pipes.",
        "Evolution by natural selection operates similarly to how selective breeding works in agriculture.",
        "The way information propagates through social networks resembles how viruses spread through populations.",
        "Quantum superposition is analogous to a coin spinning in the air before it lands on heads or tails.",
        # Technology
        "Version control for code works like track changes in a word processor, recording every modification.",
        "A database index is comparable to the index at the back of a textbook, enabling fast lookup.",
        "Machine learning training is like teaching a student through examples rather than explicit rules.",
        "Cloud computing is to local computing as renting an apartment is to owning a house.",
        "An API serves as a waiter in a restaurant, taking your request to the kitchen and bringing back the result.",
        "Just as a translator converts between languages, a compiler converts source code into machine instructions.",
        # Everyday / Social
        "Learning to ride a bicycle is analogous to learning to swim: both require practice and balance.",
        "A good teacher is to education as a good coach is to athletic performance.",
        "Budget management in a household mirrors financial management in a corporation, just at a different scale.",
        "Reading a map to navigate a city is comparable to reading documentation to navigate a codebase.",
        "Just as a garden requires regular weeding to stay healthy, a codebase needs regular refactoring.",
        "Trust in a relationship is like the foundation of a building: without it, everything collapses.",
        # Biology / Nature
        "Mitochondria are the power plants of cells, converting nutrients into energy like factories convert raw materials into electricity.",
        "The relationship between bees and flowers mirrors the relationship between service providers and customers.",
        "A coral reef ecosystem is analogous to a bustling city, with different species filling specialized roles.",
        "Just as tree roots stabilize soil, regulations stabilize financial markets.",
        # Abstract / Philosophical
        "Freedom of speech is to democracy as oxygen is to fire: necessary for it to exist.",
        "Learning from failure is like tempering steel: the heat and stress make it stronger.",
        "Patience in problem-solving is comparable to letting dough rise: rushing it ruins the result.",
        "The way memory fades over time mirrors how photographs slowly lose their color with age.",
        "A mentor's guidance is analogous to a lighthouse guiding ships safely through the fog.",
        "Just as a river carves a canyon over millennia, persistent effort shapes lasting achievements.",
        "The spread of misinformation is like the spread of an invasive species: once established, it's hard to eradicate.",
        "National borders function like cell membranes, selectively permitting or blocking the passage of people and goods.",
        "Debt accumulation works like a snowball rolling downhill, growing larger and faster over time.",
        "The balance of power between nations resembles the equilibrium between predator and prey populations.",
    ],

    ReasoningType.CONSERVATION: [
        # Physics
        "The ball was thrown upward, converting kinetic energy to potential energy, but the total mechanical energy remained constant throughout.",
        "In the closed system, heat flowed from the hot object to the cold one until both reached 50 degrees, but total thermal energy was conserved.",
        "During the collision, the two billiard balls exchanged momentum, but the total momentum before and after remained identical.",
        "The pendulum converts between kinetic and potential energy, but the total energy stays the same at every point in the swing.",
        "When the ice melted, it absorbed heat from the water, but the total energy of the ice-water system remained unchanged.",
        "Electric charge is conserved in every circuit: the current entering a junction must equal the current leaving it.",
        "In nuclear fission, mass is converted to energy, but the total mass-energy of the system is conserved per Einstein's equation.",
        "Angular momentum is conserved: when the skater pulls her arms in, she spins faster, but the product of moment and velocity stays the same.",
        "The total number of nucleons is conserved in all nuclear reactions. Protons and neutrons are neither created nor destroyed.",
        "In every elastic collision, both kinetic energy and momentum are preserved before and after impact.",
        # Chemistry
        "In the balanced chemical equation, the number of atoms on the left equals the number on the right. Mass is conserved.",
        "During combustion of methane, 1 carbon atom enters and 1 carbon atom exits as CO2. Atoms are neither created nor destroyed.",
        "The total charge in the redox reaction remains zero: electrons lost by one species are gained by another.",
        "Water electrolysis splits H2O into H2 and O2, but the total mass of products equals the total mass of reactants.",
        "In an acid-base neutralization, the total number of protons transferred equals the total number accepted.",
        # Economics / Finance
        "When money was transferred from savings to checking, the total balance across both accounts remained the same at $10,000.",
        "In a zero-sum trade, the buyer's loss equals the seller's gain. The total value in the system is unchanged.",
        "The company's total assets remained constant: cash decreased by $50K while inventory increased by $50K.",
        "Government spending is funded by taxes or debt. The total fiscal equation must balance: revenues plus borrowing equals expenditure.",
        "In a closed economy, every export is someone else's import. The global trade balance nets to zero.",
        "The accounting equation holds: assets always equal liabilities plus equity. A change on one side must balance the other.",
        # Everyday
        "After dividing the pizza into 8 slices, the total amount of pizza remained the same, just redistributed.",
        "Pouring water between three containers doesn't change the total volume. One liter is still one liter.",
        "The total number of students in the school stayed the same even though some switched from class A to class B.",
        "Rearranging furniture in the room changes the layout but the total amount of furniture remains constant.",
        "Cutting a rope into three pieces doesn't change the total length. 10 meters is still 10 meters.",
        "Converting currency from dollars to euros changes the denomination but preserves the total purchasing power at that exchange rate.",
        "The number of cards in a deck stays at 52 no matter how they are shuffled or dealt.",
        # Math / Information
        "A linear transformation preserves the dimension of the vector space. The rank before equals the rank after.",
        "In a lossless compression algorithm, all original information is preserved and can be perfectly reconstructed.",
        "Sorting an array rearranges elements but the total count and sum of elements remain unchanged.",
        "The total probability over all outcomes must equal 1. This invariant holds for every valid probability distribution.",
        "Kirchhoff's current law states that the total current entering a node equals the total current leaving. Charge is conserved.",
        "The total number of atoms is conserved in every physical process. Atoms can be rearranged but not created or destroyed.",
        "In a heat exchanger, the heat lost by the hot fluid equals the heat gained by the cold fluid. Energy is conserved.",
        "The water cycle conserves total water: evaporation, precipitation, and runoff maintain a constant global water balance.",
        "Mass is conserved during phase changes. Ice turning to water weighs exactly the same before and after.",
        "In closed-loop recycling, materials are neither created nor lost. The same aluminum is reused cycle after cycle.",
        "The fundamental theorem of calculus preserves total area: integration and differentiation are inverse operations.",
        "The baryon number is conserved in particle physics. Every reaction maintains the same count of baryons.",
        "During osmosis, water moves between compartments, but the total amount of water plus solute remains constant.",
    ],

    ReasoningType.COUNTERFACTUAL: [
        # Science
        "If the asteroid had missed Earth 66 million years ago, dinosaurs might still be the dominant species today.",
        "Had penicillin not been discovered accidentally, millions more would have died from bacterial infections.",
        "What if gravity were twice as strong? Stars would burn through their fuel much faster and life might never have evolved.",
        "If the ozone layer had not formed, ultraviolet radiation would have prevented life from colonizing land.",
        "Without the greenhouse effect, Earth's average temperature would be minus 18 degrees, making it uninhabitable.",
        "If the speed of light were infinite, causality as we understand it would break down entirely.",
        "Had the Chernobyl operators not disabled the safety systems, the meltdown could have been prevented.",
        "What if water were denser as a solid than as a liquid? Lakes would freeze from the bottom up, killing all aquatic life.",
        # History / Politics
        "If the printing press had not been invented, the Protestant Reformation might never have spread beyond Germany.",
        "Had the Allies failed on D-Day, the war in Europe could have lasted several more years.",
        "What if the internet had never been developed? Global communication would still rely on telephone and postal systems.",
        "If the Cuban Missile Crisis had escalated, nuclear war would have devastated both superpowers.",
        "Had Abraham Lincoln not been assassinated, Reconstruction might have been more successful in healing the nation.",
        "If the Berlin Wall had not fallen in 1989, German reunification would have been delayed by decades.",
        "Without the invention of the compass, maritime exploration would have been limited to coastal routes.",
        # Medicine
        "If the patient had received the correct dosage, the adverse reaction would not have occurred.",
        "Had vaccines not been developed, smallpox would likely still be claiming millions of lives annually.",
        "Without the discovery of insulin in 1921, Type 1 diabetes would remain a fatal diagnosis.",
        "If the surgeon had noticed the internal bleeding earlier, the outcome would have been different.",
        "Had the ambulance arrived five minutes later, the heart attack patient might not have survived.",
        # Economics
        "If interest rates had not been cut during the recession, unemployment would have risen even higher.",
        "Had the company invested in renewable energy sooner, they would have avoided the carbon tax penalties.",
        "What if the housing market had been properly regulated? The 2008 financial crisis might have been averted.",
        "Without government stimulus, the pandemic recession would have been significantly deeper and longer.",
        "If tariffs had not been imposed, trade between the two nations would have continued to grow.",
        # Everyday
        "If I had left five minutes earlier, I would have caught the train and arrived on time.",
        "Had she studied abroad in college, her career trajectory might have been completely different.",
        "What if electric cars had been adopted in the 1990s? Urban air quality would be much better today.",
        "If the backup generator had been maintained, the hospital would not have lost power during the storm.",
        "Without the invention of refrigeration, food distribution and preservation would be radically different.",
        "If the team had not fumbled in the fourth quarter, they would have won the championship.",
        "Had the pilot noticed the warning light, the emergency landing could have been avoided.",
        "What if we had planted trees along the riverbank? The erosion problem would have been much less severe.",
        # Technology
        "If quantum computers had been developed 20 years earlier, current encryption standards would already be obsolete.",
        "Had the software team written unit tests, the production bug would have been caught before deployment.",
        "Without open-source software, the pace of technological innovation would have been much slower.",
        "If the network had redundant connections, the single point of failure would not have caused the outage.",
        "Had the data been encrypted, the breach would not have exposed any sensitive information.",
        "What if Moore's Law had stopped in the 1990s? Smartphones as we know them would not exist.",
        "If the autonomous vehicle had detected the pedestrian earlier, the collision would have been prevented.",
        "Without standardized protocols like TCP/IP, the modern internet could never have been built.",
    ],

    ReasoningType.ABDUCTION: [
        # Science / Nature
        "The garden plants are wilting despite regular watering. The best explanation is root rot from poor drainage.",
        "The satellite lost communication for exactly 90 minutes each orbit. The most likely cause is atmospheric interference at low altitudes.",
        "Fossil seashells were found at 3000 meters elevation. This suggests the mountain was once under the ocean.",
        "The patient has a high white blood cell count and fever. The most probable diagnosis is a bacterial infection.",
        "Stars near the galaxy's edge move faster than expected. The best explanation is the presence of dark matter.",
        "Several species on separate continents share identical anatomical features. This is best explained by a common ancestor before continental drift.",
        "The lake turned green overnight. Given the recent fertilizer runoff, algal bloom is the most likely explanation.",
        "The ancient structure aligns perfectly with the summer solstice sunrise. This suggests it was designed as an astronomical observatory.",
        "The experiment yielded unexpected results that don't match any known model. The best hypothesis is an undiscovered variable affecting the outcome.",
        # Medicine / Diagnosis
        "The patient presents with joint pain, butterfly rash, and fatigue. These symptoms together are most consistent with lupus.",
        "Blood sugar levels are elevated despite medication compliance. The most likely explanation is insulin resistance developing.",
        "The child has a persistent cough, wheezing, and shortness of breath. The best diagnosis is asthma.",
        "Multiple patients in the same building developed similar symptoms. The evidence points to a shared environmental exposure.",
        "The test results show elevated liver enzymes without alcohol use. The most probable cause is a medication side effect.",
        "The patient's symptoms resolve on weekends and return on workdays. This pattern suggests an occupational exposure.",
        # Detective / Investigation
        "The door was locked from the inside with no sign of forced entry. The best explanation is the intruder had a key.",
        "Footprints in the mud lead to the window but not away. The most likely inference is the person climbed out through the window.",
        "Sales dropped 40 percent in exactly one region while all others grew. The most probable cause is the new competitor opening there.",
        "The server logs show unusual activity at 3 AM when no one was scheduled to work. This suggests unauthorized access.",
        "The fire started in three separate locations simultaneously. This evidence points to arson rather than an accident.",
        "The device failed immediately after the firmware update. The update is the most likely cause of the failure.",
        # Everyday
        "The car won't start and the lights are dim. The most probable explanation is a dead battery.",
        "The room is cold even though the thermostat reads 72 degrees. The best explanation is that the heating system has malfunctioned.",
        "There are wet footprints on the kitchen floor but no one admits being outside. Someone must have gone out without telling.",
        "The cake didn't rise properly despite following the recipe. The most likely cause is expired baking powder.",
        "My internet connection keeps dropping every evening around 8 PM. The best explanation is network congestion during peak hours.",
        "The paint is peeling only on the north wall. This is most likely caused by moisture seeping through from outside.",
        # Technology
        "The model performs well on training data but poorly on test data. The best explanation is overfitting.",
        "CPU usage spikes every hour on the hour. This suggests a scheduled cron job is consuming resources.",
        "The response time increased after the database migration. The most likely cause is missing indexes on the new database.",
        "Users in Europe report slow loading times while US users don't. This is best explained by the absence of a European CDN node.",
        "The neural network's predictions are biased toward one class. The most probable cause is class imbalance in the training data.",
        "Memory usage grows steadily until the application crashes. The best explanation is a memory leak in the code.",
        # Abstract / Reasoning
        "The prediction was wrong despite the model having high training accuracy. The most likely explanation is distribution shift between training and deployment.",
        "Two independent research groups arrived at the same result. This convergence suggests the finding is robust.",
        "The anomaly appears in data from three different instruments. Given the independent sources, it is probably a real phenomenon, not an artifact.",
        "Voter turnout was unusually high in districts with the new polling stations. The best explanation is improved accessibility.",
        "The ancient civilization collapsed rapidly. Given evidence of drought, crop failure, and social unrest, the most likely cause is a cascade triggered by climate change.",
        "The phenomenon occurs only during solar maximum years. This suggests a connection to solar magnetic activity.",
    ],

    ReasoningType.DECOMPOSITION: [
        # Math
        "To find the total cost of the trip, first calculate fuel cost from distance and mileage, then add hotel costs, then add food expenses.",
        "Solving the quadratic equation requires three steps: compute the discriminant, take its square root, and apply the quadratic formula.",
        "To evaluate the integral, first apply partial fractions to separate the terms, then integrate each fraction independently.",
        "The optimization problem breaks into two subproblems: first find the feasible region, then maximize the objective function within it.",
        "Computing the matrix determinant involves expanding along the first row and recursively computing the determinants of the submatrices.",
        "To prove the theorem, we handle two cases separately: when n is even and when n is odd.",
        "The problem reduces to three independent calculations: compute the mean, compute the variance, and construct the confidence interval.",
        # Engineering / Tech
        "Building the web application requires separate work on three components: the frontend UI, the backend API, and the database schema.",
        "Debugging the system failure means checking each layer separately: hardware, operating system, application, and network.",
        "The machine learning pipeline has four stages: data preprocessing, feature engineering, model training, and evaluation.",
        "To deploy the application, first containerize it with Docker, then configure the Kubernetes cluster, then set up the CI/CD pipeline.",
        "The performance bottleneck analysis breaks down into CPU profiling, memory profiling, and I/O profiling.",
        "Designing the microservice architecture requires decomposing the monolith into user service, payment service, and notification service.",
        "The testing strategy covers three levels: unit tests for individual functions, integration tests for components, and end-to-end tests for workflows.",
        # Science
        "Analyzing the ecosystem requires studying each trophic level independently: producers, primary consumers, secondary consumers, and decomposers.",
        "The chemical synthesis proceeds in three stages: first prepare the reagents, then run the reaction, finally purify the product.",
        "Understanding climate change requires examining three factors separately: greenhouse gas emissions, deforestation, and ocean absorption.",
        "The geological survey divides the region into four zones, each analyzed independently for mineral composition and rock type.",
        "To understand the protein's function, first determine its structure, then identify its binding sites, then test its interactions.",
        "The astronomical observation program breaks into three phases: calibration, data collection, and data reduction.",
        # Everyday / Problem Solving
        "Planning the wedding involves separate workstreams: venue and catering, guest list and invitations, and decorations and entertainment.",
        "Moving to a new city requires three major tasks: finding housing, arranging the move, and setting up utilities and services.",
        "To learn a new language, break it into components: vocabulary, grammar, pronunciation, and conversational practice.",
        "Renovating the kitchen involves separate steps: demolition, plumbing and electrical work, cabinet installation, and finishing.",
        "The project management plan divides work into four phases: initiation, planning, execution, and closure.",
        "To organize the garage, first sort items into keep, donate, and discard. Then clean the space. Finally, arrange storage.",
        "Preparing for the exam, I divided the material into three sections and allocated two days to study each one separately.",
        "The troubleshooting guide says to check each potential cause independently: power supply, cable connections, and software settings.",
        # Business / Strategy
        "The market analysis examines three segments separately: consumer demographics, competitor landscape, and regulatory environment.",
        "The business plan has four components: executive summary, market analysis, financial projections, and operational plan.",
        "To reduce costs, we analyzed each department's spending independently: manufacturing, logistics, and administration.",
        "The product launch breaks into parallel tracks: marketing campaign, supply chain preparation, and customer support training.",
        "The root cause analysis follows a systematic decomposition: who, what, when, where, why, and how.",
        # More
        "To understand the patient's condition, examine each symptom category separately: neurological, cardiovascular, and respiratory.",
        "The data pipeline decomposes into extraction from sources, transformation of formats, and loading into the warehouse.",
        "Solving the complex circuit means analyzing each loop independently using Kirchhoff's laws, then combining the results.",
        "The essay structure breaks into introduction, three body paragraphs each addressing a separate argument, and conclusion.",
        "Parallel processing divides the large dataset into chunks, processes each chunk independently, then merges the results.",
        "The security audit examines each attack surface separately: network, application, physical access, and social engineering.",
        "Understanding the economic impact requires analyzing supply-side effects, demand-side effects, and policy responses independently.",
    ],
}


def generate_expanded_traces(
    n_per_type: int = 100,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[str]]:
    """
    Generate synthetic traces from the expanded example bank.

    Samples n_per_type examples from each type's bank (with replacement if
    n_per_type exceeds bank size). Each example is a real, filled-in sentence
    that the encoder can meaningfully embed.

    Args:
        n_per_type: number of examples per reasoning type
        seed: random seed

    Returns:
        texts, labels, sources
    """
    rng = np.random.RandomState(seed)
    texts = []
    labels = []
    sources = []

    for rtype, examples in EXPANDED_EXAMPLES.items():
        if n_per_type <= len(examples):
            selected_idx = rng.choice(len(examples), size=n_per_type, replace=False)
        else:
            # Sample with replacement if requesting more than available
            selected_idx = rng.choice(len(examples), size=n_per_type, replace=True)

        for idx in selected_idx:
            texts.append(examples[idx])
            labels.append(int(rtype))
            sources.append("synthetic_expanded")

    # Shuffle
    order = rng.permutation(len(texts))
    texts = [texts[i] for i in order]
    labels = [labels[i] for i in order]
    sources = [sources[i] for i in order]

    return texts, labels, sources


# ─── Benchmark-Derived Traces ──────────────────────────────────────────────────

def collect_benchmark_traces(
    benchmarks: List[str] = None,
    max_per_benchmark: int = 2000,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[Dict], List[str]]:
    """
    Collect reasoning traces from benchmarks and label with heuristic labeler.

    For each benchmark question, concatenate question + correct answer and
    label with the heuristic labeler. Returns soft labels.
    """
    if benchmarks is None:
        benchmarks = ['gsm8k', 'arc_challenge', 'strategyqa', 'folio']

    from shared.data_utils import LOADERS
    import inspect

    labeler = HeuristicLabeler()
    rng = np.random.RandomState(seed)

    texts = []
    hard_labels = []
    soft_labels = []
    sources = []

    for benchmark in benchmarks:
        if benchmark not in LOADERS:
            logger.warning(f"Unknown benchmark: {benchmark}, skipping")
            continue

        try:
            loader_fn = LOADERS[benchmark]
            kwargs = {'split': 'train'}
            if 'seed' in inspect.signature(loader_fn).parameters:
                kwargs['seed'] = seed
            questions, choices, labels_list = loader_fn(**kwargs)
        except Exception as e:
            logger.warning(f"Failed to load {benchmark}: {e}")
            continue

        # Sample subset (larger cap now)
        n_available = len(questions)
        n_sample = min(n_available, max_per_benchmark)
        indices = rng.permutation(n_available)[:n_sample]

        for idx in indices:
            q = questions[idx]
            correct_choice = choices[idx][labels_list[idx]]
            text = f"{q} {correct_choice}"

            soft = labeler.label(text)
            hard, conf = labeler.label_hard(text)

            texts.append(text)
            hard_labels.append(int(hard))
            soft_labels.append(soft)
            sources.append(f"benchmark_{benchmark}")

    return texts, hard_labels, soft_labels, sources


def class_balanced_sample(
    texts: List[str],
    labels: List[int],
    soft_labels: List[Dict],
    sources: List[str],
    target_per_type: int = 500,
    seed: int = 42,
) -> Tuple[List[str], List[int], List[Dict], List[str]]:
    """
    Perform class-balanced sampling: cap over-represented types,
    oversample under-represented types.

    Args:
        texts, labels, soft_labels, sources: input data
        target_per_type: target number of samples per reasoning type
        seed: random seed

    Returns:
        Balanced texts, labels, soft_labels, sources
    """
    rng = np.random.RandomState(seed)

    # Group by label
    type_indices = {int(rt): [] for rt in ReasoningType}
    for i, label in enumerate(labels):
        type_indices[label].append(i)

    balanced_indices = []
    for rt in ReasoningType:
        rt_int = int(rt)
        indices = type_indices[rt_int]
        n_available = len(indices)

        if n_available == 0:
            logger.warning(f"  {REASONING_TYPES[rt].short_name}: 0 samples, skipping")
            continue

        if n_available >= target_per_type:
            # Downsample
            selected = rng.choice(indices, size=target_per_type, replace=False)
        else:
            # Oversample (keep all originals + sample extras with replacement)
            extra_needed = target_per_type - n_available
            extras = rng.choice(indices, size=extra_needed, replace=True)
            selected = np.concatenate([indices, extras])

        balanced_indices.extend(selected.tolist())
        logger.info(
            f"  {REASONING_TYPES[rt].short_name:>8}: "
            f"{n_available} available -> {len(selected)} sampled"
        )

    rng.shuffle(balanced_indices)

    out_texts = [texts[i] for i in balanced_indices]
    out_labels = [labels[i] for i in balanced_indices]
    out_soft = [soft_labels[i] for i in balanced_indices]
    out_sources = [sources[i] for i in balanced_indices]

    return out_texts, out_labels, out_soft, out_sources


# ─── Full Pipeline ─────────────────────────────────────────────────────────────

def build_training_dataset(
    encoder,
    n_synthetic_per_type: int = 50,
    benchmarks: List[str] = None,
    max_per_benchmark: int = 2000,
    target_per_type: int = 750,
    balance: bool = True,
    seed: int = 42,
    batch_size: int = 64,
) -> ReasoningTraceDataset:
    """
    Build the complete ARTI training dataset from all sources.

    Combines:
    1. Built-in examples (24 samples, ground truth labels)
    2. Expanded synthetic sentences (n_synthetic_per_type * 8 samples)
    3. Benchmark traces (class-balanced after heuristic labeling)

    Args:
        encoder: SentenceTransformerEncoder for embedding texts
        n_synthetic_per_type: expanded synthetic examples per reasoning type
        benchmarks: which benchmarks to use
        max_per_benchmark: max examples per benchmark (before balancing)
        target_per_type: target samples per type after balancing
        balance: whether to apply class-balanced sampling
        seed: random seed
        batch_size: encoding batch size

    Returns:
        ReasoningTraceDataset ready for training
    """
    all_texts = []
    all_labels = []
    all_soft_labels = []
    all_sources = []

    # Source 1: Built-in examples (ground truth)
    texts, labels, sources = collect_builtin_examples()
    all_texts.extend(texts)
    all_labels.extend(labels)
    for label in labels:
        soft = {rt: 0.0 for rt in ReasoningType}
        soft[ReasoningType(label)] = 1.0
        all_soft_labels.append(soft)
    all_sources.extend(sources)
    logger.info(f"Built-in examples: {len(texts)}")

    # Source 2: Expanded synthetic sentences (ground truth)
    texts, labels, sources = generate_expanded_traces(n_synthetic_per_type, seed)
    all_texts.extend(texts)
    all_labels.extend(labels)
    for label in labels:
        soft = {rt: 0.0 for rt in ReasoningType}
        soft[ReasoningType(label)] = 1.0
        all_soft_labels.append(soft)
    all_sources.extend(sources)
    logger.info(f"Expanded synthetic traces: {len(texts)}")

    # Source 3: Benchmark traces (heuristic labeled)
    benchmark_texts = []
    benchmark_labels = []
    benchmark_soft = []
    benchmark_sources = []

    try:
        bm_texts, bm_labels, bm_soft, bm_sources = collect_benchmark_traces(
            benchmarks=benchmarks,
            max_per_benchmark=max_per_benchmark,
            seed=seed,
        )
        benchmark_texts = bm_texts
        benchmark_labels = bm_labels
        benchmark_soft = bm_soft
        benchmark_sources = bm_sources
        logger.info(f"Raw benchmark traces: {len(bm_texts)}")

        # Show raw distribution before balancing
        raw_dist = Counter(bm_labels)
        for rtype in ReasoningType:
            count = raw_dist.get(int(rtype), 0)
            pct = count / max(len(bm_labels), 1) * 100
            logger.info(f"  {REASONING_TYPES[rtype].short_name:>8}: {count} ({pct:.1f}%)")

    except Exception as e:
        logger.warning(f"Benchmark trace collection failed: {e}")

    # Apply class-balanced sampling to benchmarks
    if balance and len(benchmark_texts) > 0:
        logger.info(f"\nApplying class-balanced sampling (target={target_per_type}/type)...")

        # Compute how many benchmark samples we need per type
        # accounting for synthetic samples already collected
        synthetic_dist = Counter(all_labels)
        benchmark_target = {}
        for rt in ReasoningType:
            synthetic_count = synthetic_dist.get(int(rt), 0)
            remaining = max(0, target_per_type - synthetic_count)
            benchmark_target[int(rt)] = remaining

        logger.info("Benchmark targets (after accounting for synthetic):")
        for rt in ReasoningType:
            logger.info(
                f"  {REASONING_TYPES[rt].short_name:>8}: "
                f"synthetic={synthetic_dist.get(int(rt), 0)}, "
                f"need={benchmark_target[int(rt)]}"
            )

        # Class-balanced sampling from benchmarks
        bm_texts_bal, bm_labels_bal, bm_soft_bal, bm_sources_bal = (
            class_balanced_sample(
                benchmark_texts, benchmark_labels, benchmark_soft,
                benchmark_sources,
                target_per_type=max(benchmark_target.values()),
                seed=seed,
            )
        )

        all_texts.extend(bm_texts_bal)
        all_labels.extend(bm_labels_bal)
        all_soft_labels.extend(bm_soft_bal)
        all_sources.extend(bm_sources_bal)
        logger.info(f"Balanced benchmark traces: {len(bm_texts_bal)}")
    elif len(benchmark_texts) > 0:
        all_texts.extend(benchmark_texts)
        all_labels.extend(benchmark_labels)
        all_soft_labels.extend(benchmark_soft)
        all_sources.extend(benchmark_sources)

    logger.info(f"\nTotal training samples: {len(all_texts)}")

    # Final label distribution
    dist = Counter(all_labels)
    for rtype in ReasoningType:
        count = dist.get(int(rtype), 0)
        pct = count / max(len(all_labels), 1) * 100
        logger.info(f"  {REASONING_TYPES[rtype].short_name:>8}: {count} ({pct:.1f}%)")

    return ReasoningTraceDataset.from_texts(
        texts=all_texts,
        labels=all_labels,
        encoder=encoder,
        soft_labels=all_soft_labels,
        sources=all_sources,
        batch_size=batch_size,
    )


def compute_class_weights(dataset: ReasoningTraceDataset) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for balanced loss.

    Returns:
        Tensor of shape [N_REASONING_TYPES] with weights
    """
    counts = Counter(dataset.labels.numpy().tolist())
    total = len(dataset)
    weights = torch.zeros(N_REASONING_TYPES)

    for rt in ReasoningType:
        rt_int = int(rt)
        count = counts.get(rt_int, 0)
        if count > 0:
            # Inverse frequency, normalized so mean weight = 1
            weights[rt_int] = total / (N_REASONING_TYPES * count)
        else:
            weights[rt_int] = 1.0

    return weights


def save_dataset(dataset: ReasoningTraceDataset, path: str):
    """Save dataset to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'embeddings': dataset.embeddings,
        'labels': dataset.labels,
        'soft_labels': dataset.soft_labels,
        'texts': dataset.texts,
        'sources': dataset.sources,
    }, path)
    logger.info(f"Dataset saved to {path} ({len(dataset)} samples)")


def load_dataset(path: str) -> ReasoningTraceDataset:
    """Load dataset from disk."""
    data = torch.load(path, weights_only=False)
    return ReasoningTraceDataset(
        embeddings=data['embeddings'],
        labels=data['labels'],
        soft_labels=data.get('soft_labels'),
        texts=data.get('texts', []),
        sources=data.get('sources', []),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing ARTI data collection (v2 - scaled)...")

    # Test expanded examples
    for rtype, examples in EXPANDED_EXAMPLES.items():
        print(f"  {REASONING_TYPES[rtype].short_name:>8}: {len(examples)} examples")
    total_expanded = sum(len(v) for v in EXPANDED_EXAMPLES.values())
    print(f"  Total expanded examples: {total_expanded}")

    # Test built-in examples
    texts, labels, sources = collect_builtin_examples()
    print(f"\nBuilt-in examples: {len(texts)}")

    # Test expanded generation
    texts, labels, sources = generate_expanded_traces(n_per_type=40)
    print(f"Expanded traces (40/type): {len(texts)}")
    dist = Counter(labels)
    for rtype in ReasoningType:
        print(f"  {REASONING_TYPES[rtype].short_name:>8}: {dist.get(int(rtype), 0)}")

    # Test heuristic labeling
    labeler = HeuristicLabeler()
    test = "Because the temperature dropped, the pipes froze. Therefore we had water damage."
    hard, conf = labeler.label_hard(test)
    print(f"\nHeuristic test: {REASONING_TYPES[hard].name} ({conf:.0%})")

    print("\nData collection tests passed!")
