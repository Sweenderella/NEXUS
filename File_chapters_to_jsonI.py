#for chapter 5
import re
import json


def parse_pdf_structure_with_page_numbers(pdf_text,output_json_path):
    # Initialize structure for chapters
    chapters = []
    current_chapter = None
    current_section = None
    current_subsection = None
    current_page = 1  # Assuming page numbers start from 1, can adjust based on actual data
    current_chapter_number = None  # Initialize the chapter number
    current_section_number = None  # Initialize the section number
    subsection_number = None  # Initialize the subsection number

    # Split the text into lines for processing
    lines = pdf_text.splitlines()

    for line in lines:
        # Check if page number is specified in the line, e.g., "4-60" (chapter 4, page 60)
        page_match = re.match(r'(\d+)-(\d+)', line)
        if page_match:
            current_chapter_number = page_match.group(1)  # Extract chapter number (e.g., '4')
            current_page = int(page_match.group(2))  # Extract page number (e.g., '60')
            continue

        # Check for chapter titles (e.g., "Chapter 4: Consumer Impacts")
        #chapter_match = re.match(r'^(Chapter \d+:\s.*)', line)
        chapter_match = re.match(r'Chapter (\d+):\s*(.*)', line)
        if chapter_match:
            if current_chapter:
                chapters.append(current_chapter)
            current_chapter = {
                "chapter_title": chapter_match.group(2),
                "chapter_number": chapter_match.group(1),  # Include chapter number
                "page_number": current_page,
                "sections": []
            }
            current_section = None  # Reset the section for each new chapter
            continue

        # Check for section titles (e.g., "4.1 Modeling the Purchase Decision")
        section_match = re.match(r'(\d+\.\d+(?:\.\d+)*)(?:\s+)(.*)', line)
       #updating here section_match = re.match(r'(\d+\.\d+(?:\.\d+)*): (.+)', line)
        if section_match:
            current_section_number = section_match.group(1)  # Capture section number (e.g., '4.5')
            section_title = section_match.group(2)  # Capture section title

            if current_section:
                current_chapter["sections"].append(current_section)
            current_section = {
                "section_number": current_section_number,  # Add section number
                "title": section_title,
                "page_number": current_page,
                "subsections": [],
                "content": ""
            }
            continue

        # Check for subsection titles (e.g., "4.1.1 Costs Incorporated in the Purchase Decision")
        subsection_match = re.match(r'^\d+\.\d+\.\d+\s+(.*)', line)
        if subsection_match:
            # Ensure current_section is not None before adding subsections
            if current_section is None:
                current_section = {
                    "section_number": current_section_number,
                    "title": "Untitled Section",  # Default placeholder if no section was initialized
                    "page_number": current_page,
                    "subsections": [],
                    "content": ""
                }

            # Use the most recent section number for the subsection
            subsection_number = subsection_match.group(0).split(' ')[0]

            # Add the subsection to the section
            if current_subsection:
                current_section["subsections"].append(current_subsection)

            current_subsection = {
                "subsection_number": subsection_number,  # Include subsection number
                "title": subsection_match.group(1),
                "page_number": current_page,
                "content": ""
            }
            continue

        # Append content to the current section or subsection
        if current_subsection:
            current_subsection["content"] += " " + line.strip()
        elif current_section:
            current_section["content"] += " " + line.strip()

    # Add the last section/subsection if present
    if current_subsection:
        current_section["subsections"].append(current_subsection)
    if current_section:
        current_chapter["sections"].append(current_section)
    if current_chapter:
        chapters.append(current_chapter)
    
    # Save the JSON structure to the output file
    with open(output_json_path, 'w', encoding='utf-8') as json_file:
        json.dump(chapters, json_file, indent=4)
        print(f"JSON file saved to {output_json_path}")

    return {"chapters": chapters}



# Example PDF text extraction (as a string)
pdf_text = """
Chapter 5: Electric Infrastructure Impacts
As plug-in electric vehicles (PEVs) are projected to represent a significant share of the future
U.S. light- and medium-duty vehicle fleet, EPA has developed new approaches to estimate the
power sector emission implications (i.e., from electricity generation, transmission, and
distribution system, which typically ends at a service drop; the run of cables from the electric
power utility's distribution power lines to the point of connection to a customer's premises) of
increased PEV charging. EPA combined the use of three analytical tools to incorporate gridrelated
emissions from PEV charging demand within the light- and medium-duty vehicle
emissions inventory analysis for the proposal:
1) OMEGA
2) A suite of electric vehicle infrastructure modeling tools (EVI-X) developed by the National
Renewable Energy Laboratory (NREL)
3) The Integrated Planning Model (IPM)
Chapter 5.1 below provides a summary of EVI-X and how these tools were used together with
OMEGA to estimate charge demand inputs for IPM. The IPM modeling results and how the
results were incorporated into the emissions inventory analysis are described in Chapters 5-8 and
Chapter 9. Chapter 5.3 describes our assessment of PEV charging infrastructure. It should be
noted that charging infrastructure is different from the electric power utility distribution system
infrastructure, which is comprised of distribution feeder circuits, switches, protective equipment,
primary circuits, distribution transformers, secondaries, service drops, etc. The electric power
utility distribution system infrastructure typically ends at a service drop (i.e. the run of cables
from the electric power utility's distribution power lines to the point of connection to a
customer's premises).
Finally, the potential impacts on pending changes to the power sector on grid resiliency are
discussed in Chapter 5.4.
5.1 Modeling PEV Charge Demand and Regional Distribution
Under an Interagency Agreement between EPA and the U.S. Department of Energy, NREL
has continued its development of a suite of electric vehicle infrastructure modeling tools (EVI-X)
and methods for simulating PEV charging infrastructure requirements and associated electricity
loads from best available data. EVI-X tools have informed multiple national, state, and local
PEV charging infrastructure planning studies (E. Wood, et al. 2017) (E. Wood, C. Rames, et al.
2018) (Alexander, et al. 2021), including a forthcoming national infrastructure assessment
through 2030 (Wood, Borlaug, et al. 2023). As noted above, this infrastructure differs from that
of electric power utility distribution system infrastructure. Within the emissions inventory
analysis for the proposal, EVI-X models are used to translate scenario-specific forecasts of
national light-duty vehicle stock and annual energy consumption from the OMEGA model into
spatially disaggregated hourly load profiles required for subsequent power sector modeling using
the Integrated Planning Model (IPM) . The primary components of the process
flow from OMEGA outputs to IPM inputs . IPM outputs also flow back
into inventory analyses in OMEGA as PEV emissions factors
5-1
OMEGA’s national PEV stock projections and PEV attributes into hourly load profiles.
5.1.1 PEV Disaggregation and Charging Simulation
As described in further detail in Chapter 2 of the DRIA, the OMEGA model evaluates the cost
of compliance for meeting the standards and options analyzed within the proposed rule. Each
OMEGA run produces scenario-specific projections of national vehicle sales, stock, energy
consumption, and tailpipe emissions. For PEVs, however, tailpipe emissions are zero in the case
of battery electric vehicles (BEVs) and during the charge-depleting operation of plug-in hybrid
electric (PHEVs) with resulting emissions occuring upstream at the electricity generation source,
thus expanding the requisite analytical boundaries of the system with respect to determination of
emissions inventory impacts. To produce estimates of the spatiotemporal charging loads needed
for power sector emissions modeling, the national PEV stock from OMEGA must first be
disaggregated regionally.
The framework developed for PEV disaggregation leverages a likely adopter model (LAM)
adapted by NREL (Ge, et al. 2021) to rank vehicles in the private light-duty fleet for their
likelihood to be replaced by a PEV based on publicly available demographic data, including
housing type, income, tenure (rent or own), state policies (ZEV states), and population density.
The model is trained on the revealed preferences of 3,772 survey respondents (228 PEV owners)
across the United States as described in (Ge, et al. 2021). Vehicle registration data from June
2022 (Experian 2022) were used to develop a set of chassis-specific LAMs for disaggregating
PEV sedans, S/CUVs, pickups, and vans based on current regional vehicle type preferences. This
process is outlined in Figure 2.
5-2
Figure 5-2: Procedure for disaggregating OMEGA national PEV stock projections to IPM
regions.
Vehicles modeled within OMEGA were first assigned to a simplified chassis type (i.e., sedan,
S/CUV, pickup, van). Next, the total number of vehicles in the simplified chassis types were
used as inputs to each of the four chassis-specific LAMs to disaggregate PEVs into IPM regions
based on regional vehicle type preferences and the likelihood of PEV adoption.
The OMEGA model generates vehicle adoption projections for thousands of unique PEV
models over time. Conducting detailed charging simulations for each of these models would be
computationally prohibitive and produce results not expected to meaningfully differentiate from
those generated by a reduced set of representative PEV models. Thus, a clustering approach was
used to generate these representative PEV models for simulation from the complete set of
OMEGA vehicles. K-means clustering was performed over each PEV’s respective battery
capacity (kWh) and energy consumption rate (kWh/mi.) parameters as specified by OMEGA. A
silhouette analysis was used to determine the appropriate number of clusters (k=6 for BEVs, k=2
for PHEVs) and OMEGA vehicles were assigned to clusters that minimize the Euclidean
distance to the centroids of the two normalized (Z-score) parameters. These assignments were
retained and used to map OMEGA vehicles to the most similar synthetic representative PEV
model. The cluster centroids were used to produce the battery capacity and energy consumption
rate parameters for the eight representative PEVs required for subsequent PEV charging
simulations. An additional parameter, the max DC charge acceptance, was defined as the
maximum effective charging rate over a typical 20percent to 80percent SOC DC fast charge
(DCFC) window. This was required to simulate DCFC for BEVs and was not directly specified
by the OMEGA model. PHEVs were assumed to be incapable of using DCFC equipment. For
modeling BEV DCFC, a simple heuristic was applied such that pre-2030 model years (Gen 1
batteries) would be capable of 1.5C charging while model year 2030 and after BEVs would be
capable of charging at 3C (Gen 2 batteries).89 The key parameters for simulating charging for
each of the representative PEVs are shown in Table 5-1.
Three separate EVI-X models developed by NREL, namely EVI-Pro (for typical daily travel),
EVI-RoadTrip (for long-distance travel), and EVI-OnDemand (for ride-hailing applications)
89 C-rate is a measure of the rate at which a battery is charged/discharged relative to its maximum energy storage
capacity. For example, 1.5C indicates that the battery is fully charged in 40 minutes, while 3C indicates a full
charge in 20 minutes
5-3
were used to estimate composite PEV charging load profiles under a unified set of assumptions:
PEV fleet composition, regional home charging access (Ge, et al. 2021), regional weather
conditions, public/workplace infrastructure availability, and charging preferences.

Figure 5-3 shows a schematic summary of the EVI-X models. The EVI-X models perform
bottom-up simulations of charging behavior by superimposing the use of a PEV over travel data
from internal combustion engine vehicles. These independent, but coordinated, simulations
produce daily charging demands for typical PEV use, long-distance travel, and ride-hailing
electrification, respectively, which are indexed in time (hourly over a representative 24-hr period
for weekdays and weekends) and space (county). This process is shown in Figure 3 and
described in (Wood, Borlaug, et al. 2023).
Figure 5-3: EVI-X National light-duty vehicle framework simulation showing
spatiotemporal energy demands for three separate use cases: typical daily travel (EVI-Pro),
long-distance travel (EVI-RoadTrip), and ride-hailing (EVI-OnDemand).
5-4
Following the PEV charging simulations, load profiles were aggregated from the county-level
into IPM regions and converted from local time to Eastern Standard Time (EST) for IPM
implementation. A final corrective step was taken to ensure that the annual energy consumption
estimates supplied by OMEGA were reflected in the PEV load profiles.
For a given OMEGA national PEV stock projection file, the modeling framework produces a
typical weekday and weekend 24-hour (EST) load profile for all IPM regions (plus Hawaii,
Alaska, and Puerto Rico) and analysis years (2026, 2028, 2030, 2032, 2035, 2040, 2045, 2050,
2055). Load profiles were analyzed using output from four separate OMEGA analytical cases:
1) No-action Case: Vehicle electrification under the existing 2023 through 2026 light-duty
vehicle GHG standards as represented by the standards finalized by EPA December 30, 2021
(86 FR 74434 2021), with updated OMEGA compliance modeling (see DRIA Chapter 2).
2) Action Case: Proposed light-and medium-duty vehicle standards
3) High BEV Sensitivity Case
4) Low BEV Sensitivity Case
These analytical cases are described in more detail below. Figure 5-4 provides an example of
how specific load profiles may be used to infer annual PEV charging demands for 2030 and 2050
using an example OMEGA analytical scenario (the "Action Case").
5-5
Figure 5-4: Annual PEV charging loads (2030 and 2050 are shown) for each IPM region in
the contiguous United States based on OMEGA charge demand for the proposal in 2030
(top) and 2050 (bottom).
5-6
In addition to the total hourly energy demands for PEV charging, energy demands were also
broken out by the following charger types – home Level 1 (L1), home Level 2 (L2), work L2,
public L2, and public DCFC (Figure 5-5). See section 5.3.1.2. for additional discussion. Note
that these have been converted to EST and reflect an unmanaged charging scenario where drivers
do not prioritize charging at certain times of the day (i.e., charging starts as soon as possible
when vehicles are plugged in without consideration of electricity price or other factors).
In Figure 5-5, there are clear differences in the magnitude, shape, and charger types between
the West Texas (left–ERC_WEST, containing mostly rural areas and small cities such as
Midland and Odessa) and East Texas (right–ERC_REST, including multiple major population
centers such as Houston, San Antonio, Austin, and Dallas-Ft. Worth) regions. The EVI-X
National light-duty vehicle framework conducts charging simulations that are reflective of the
regional differences in EV adoption, vehicle type preferences, home ownership, weather
conditions, and travel patterns. These demonstrative results reflect how in ERC_WEST, EV
adoption is projected to be low (due to limited population and revealed vehicle preferences)
leading to a reduced demand for home-based charging while public DCFC demands for longdistance
travel across the region (e.g., road trips) are amplified. This leads to a disproportionate
share of public DCFC charging demand along highway corridors within the ERC_WEST region.
Alternatively, simulated charging demands in the ERC_REST are dominated by home and
workplace charging due to the higher EV adoption and urban travel patterns more common to the
region.
The OMEGA national PEV outputs and the resulting regionalized IPM inputs from EVI-X for
each of the four analyzed cases, for each IPM region and all analytical years (2026, 2028, 2030,
2032, 2035, 2040, 2045, 2050, 2055) are summarized within a separate PEV Regionalized
Charge Demand Report (McDonald 2023).
5-7
Figure 5-5: Yearly hourly (in EST) weekday and weekend load profiles for two IPM
regions (ERC_WEST, west Texas; and ERC_REST, east Texas) broken out by charger
type for an example OMEGA analytical scenario.
5.2 Electric Power Sector Modeling
The analyses for the proposal used EPA's Power Sector Modeling Platform, which utilizes the
Integrated Planning Model (IPM). IPM is a multi-regional, dynamic, deterministic linear
programming model of the U.S. electric power sector. It provides projections of least-cost
capacity expansion, electricity dispatch, and emission control strategies for meeting energy
demand and environmental, transmission, dispatch, and reliability constraints. IPM can be used
to evaluate the cost and emissions impacts of proposed policies to limit emissions of sulfur
5-8
dioxide (SO2), nitrogen oxides (NOX), carbon dioxide (CO2), hydrogen chloride (HCl), and
mercury (Hg) from the electric power sector. Post-processing IPM outputs allows for the
processing of other emissions, such as volatile organic compounds (VOC) and non-CO2 GHGs.
The power-sector modeling used for the proposal included power-sector-related provisions of
both the Bipartisan Infrastructure Law (BIL) and the Inflation Reduction Act (IRA). Additional
information regarding power-sector modeling is available via a report submitted to the docket
(U.S. EPA 2023).
5.2.1 Estimating Retail Electricity Prices
The Retail Price Model (RPM) was developed to estimate retail prices of electricity using
wholesale electricity prices generated by the IPM. The RPM provides a first-order estimate of
average retail electricity prices using information from EPA’s Power Sector Modeling Platform
v6.21 using the Integrated Planning Model (IPM) and the EIA’s Annual Energy Outlook (AEO).
This model was developed by ICF a under contract with EPA (ICF 2019).
IPM includes a wholesale electric power market model that projects wholesale prices paid to
generators. Electricity consumers—industrial, commercial, and residential customers—face a
retail price for electricity that is higher than the wholesale price because it includes the cost of
wholesale power and the costs of transmitting and distributing electricity to end-use consumers.
The RPM was developed to estimate retail prices of electricity based on outputs of EPA’s Base
Case using IPM and a range of other assumptions, including the method of regulation and pricesetting
in each state. Traditionally, cost-of-service (COS) or Rate-of-Return regulation sets rates
based on the estimated average costs of providing electricity to customers plus a “fair and
equitable return” to the utility’s investors. States that impose cost-of-service regulation typically
have one or more investor-owned utilities (IOUs), which own and operate their own generation,
transmission, and distribution assets. They are also the retail service provider for their franchised
service territory in which IOUs operates. Under this regulatory structure, retail power prices are
based on average historical costs and are established for each class of service by state regulators
during periodic rate case proceedings. Additional documentation on the RPM can be found at on
the EPA website.
5.2.2 IPM emissions post-processing
Emissions of non-CO2 GHG (methane, nitrous oxide), PM, VOC, CO and NH3 were
calculated via post-processing of IPM power sector data and using EPA-defined emissions
factors. The EPA GHG Emissions Factors Hub was used to determine fuel-specific emissions
factors for methane and nitrous oxide emissions for the electric power sector (U.S. EPA 2022a).
Emissions factors used for post-processing of PM, VOC, CO and NH3 were documented as part
of EPA’s Power Sector Modeling Platform v6 - Summer 2021 Reference Case (U.S. EPA 2021).
5.2.3 IPM National-level Demand, Generation, Emissions and Costs
As EPA was in the process of developing this proposal in the fall of 2022, EPA's Clean Air
Markets Division (CAMD) completed an initial power sector modeling analysis of the BIL and
IRA. The IRA provisions modeled within IPM included:
• Clean Electricity Production and Investment Tax Credits
• Existing Nuclear Production Tax Credit
5-9
• Carbon Capture and Storage 45Q Tax Credit
This initial modeling did not include other power sector impacts, such as demand impacts
from electrification and energy efficiency provisions, however these are likely to be part of
future CAMD power sector analyses.
The initial modeling of the IRA showed a 70percent reduction of power sector related CO2
emissions from current levels by 2055, and that the changes in CO2 emissions would be driven
primarily by increases in renewable generation and enabled by increased use of grid battery
storage capacity 
(BIL) and the Inflation Reduction Act (IRA)
Similar to CAMD's earlier power sector analysis, the power sector analysis for both the
proposal and a no-action case show significant reductions in CO2 emissions from 2028 through
2050 despite increased generation and largely due to increased use of renewables for generation.
5-10
A summary of national electric power sector emissions, demand, generation, and cost for the
no-action case and for the proposal are presented in Table 5-2 and Table 5-3, respectively. Note
that the total costs presented in both tables represent:
• Capital costs for building new power plants as well as retrofits
• Variable and fixed operation and maintenance costs
• Fuel costs
• Cost of transporting and storing CO2
Proposal
No Action
2025 2030 2035 2040 2045 2050
Year
5-13
5-14
5.2.4 Retail Price Modeling Results
EPA estimated the change in the retail price of electricity (2020$) using the Retail Price
Model (RPM) and using the same methodology used in recent power-sector rulemakings (U.S.
EPA 2022b). The RPM was developed by ICF for EPA (ICF 2019) and uses the IPM estimates
of changes in the cost of generating electricity to estimate the changes in average retail electricity
prices. The prices are average prices over consumer classes (i.e., consumer, commercial, and
industrial) and regions, weighted by the amount of electricity used by each class and in each
region. The RPM combines the IPM annual cost estimates in each of the 74 IPM regions with
EIA electricity market data for each of the 25 NERC/ISO90 electricity supply subregions (Table
5-4 and Figure 5-12) in the electricity market module of the National Energy Modeling System
(NEMS) (U.S. Energy Information Administraton 2019). Table 5-4 summarizes the projected
percentage changes in the retail price of electricity for the proposal versus a no-action case,
respectively. Consistent with other projected impacts presented above, average retail electricity
price differences at the national level are projected to be small at less than 1percent difference in
2030 and 2050. Regional average retail electricity price differences showed small increases or
decreases (less than approximately 1 to 2percent) with the sole exception of PJMC, which is the
PJM Commonwealth Edison (Metropolitan Chicago) NERC/ISO subregion.
There is a general trend of reduced national average retail electricity prices from 2021 through
2050, which is largely due to reduced fuel costs from increased use of renewables for generation.
Figure 5-12: Electricity Market Module Regions (U.S. Energy Information Administraton
2019).
90 NERC is the National Electricity Reliability Corporation. ISO is an Independent System Operator, sometimes
referred to as a Regional Transmission Organization.
5-15
5-16
Table Notes:
*Percentage increase in average retail electricity price from the Proposal to a no-action case. Negative percentages reflect a decrease in average retail electricity price for
the proposal.
†One mill is equal to 1/1,000 U.S. dollar, or 1/10 U.S. cent. 2020 mills per kilowatthour (mills/kWh) are equivalent to 2020 dollars per megawatt-hour ($/MWh)
5-17
5.2.5 New Builds, Retrofits and Retirements of EGUs
The electric power sector emissions modeling undertaken in support of this rulemaking, using
IPM (described at the beginning of Chapter 5.2), also projects the anticipated mix of electric
power plants required to meet the imposed electric power load from vehicle electrification,
subject to various constraints. These power plants are referred to here collectively as Electric
Generating Units (EGU). This definition includes all types of generating facilities (e.g. fossil
fuel-fired combustion, nuclear, hydroelectric, renewable, etc.).
This modeling reveals anticipated EGU retirements, EGU retrofits, and new EGU
construction, which are discussed below. EGUs are retired by IPM when announced by their
owner and for economic reasons. The IRA and BIL resulted in many EGU retirements. As such,
the number and types of EGU retirements associated with the proposed rule when compared to a
no-action case are small in comparison to those retirements that occurred as a result of the IRA
and BIL.
New EGU capacity modelled by IPM for the no-action case is summarized in Table 5-6. New
EGU capacity modelled by IPM for the proposal is summarized in Table 5-7. EGU retirements
modelled by IPM for the no-action and for the proposal are summarized in , respectively. Incremental EGU retirements and incremental new modeled EGU capacity are
summarized in Table 5-10 and Table 5-11, respectively.
For the no-action case, the retirement of coal-fired EGUs account for 81.1%, 80.4%, 75.7%,
74.7%, 65.3%, and 57.4% of all EGU retirements for 2028, 2030, 2035, 2040, 2045, and 2050,
respectively (see Table 5-8). For the proposal, the retirement of coal-fired EGUs are very similar
to the no-action case at 81.7%, 81.3%, 76.2%, 75.7%, 66.0%, and 57.8% of all EGU retirements
for 2028, 2030, 2035, 2040, 2045, and 2050, respectively (see Table 5-9).
For the no-action case, cumulative power generation from new solar EGU builds are expected
account for 11.3%, 23.2%, 28.9%, 31.5%, 28.7%, and 29.2% of all new power generation for
2028, 2030, 2035, 2040, 2045, and 2050, respectively. Also, cumulative power generation from
new wind-powered EGU builds are expected account for 27.0%, 36.9%, 45.4%, 42.7%, 42.9%,
and 40.1% of all new power generation for 2028, 2030, 2035, 2040, 2045, and 2050,
respectively. Likewise, cumulative power generation from new energy storage EGU builds are
expected account for 31.8%, 24.4%, 15.7%, 13.0%, 10.3%, and 9.2% of all new power
generation for 2028, 2030, 2035, 2040, 2045, and 2050, respectively.
New generation for the proposal is similar to the no-action case. For the proposal,
cumulative power generation from new solar EGU builds are expected account for 10.9%,
22.8%, 28.8%, 31.8%, 29.2%, and 29.5% of all new power generation for 2028, 2030, 2035,
2040, 2045, and 2050, respectively. Also, cumulative power generation from new wind-powered
EGU builds are expected account for 27.3%, 36.7%, 44.1%, 41.6%, 41.6%, and 39.4% of all new
power generation for 2028, 2030, 2035, 2040, 2045, and 2050, respectively. Likewise,
cumulative power generation from new energy storage EGU builds are expected account for
31.0%, 24.7%,15.8%,12.4%, 9.7%, and 8.8% of all new power generation for 2028, 2030, 2035,
2040, 2045, and 2050, respectively.
Solar-power is expected to become the single largest new source of EGU capacity for 2040,
2045, and 2050, accounting for 34.4%, 35.4%, and 34.0% of overall new EGU capacity,
5-18
respectively. Wind-driven EGUs are expected to comprise the second largest new source of EGU
capacity for 2040 and 2050, accounting for 28.5% and 28.2% of overall new EGU capacity,
respectively.
5-20
When comparing the proposal to the no-action case, only existing coal-fired EGUs were
found to receive retrofits. The cumulative capacity modeled by IPM totaled to 1,994.4 MW,
1,891.4 MW, 10,554.4 MW, 3,745.3 MW, 848.5 MW and 2,047.3 MW for the model run years
of 2028, 2030, 2035, 2040, 2045, and 2050, respectively.
5.2.6 Interregional Dispatch
IPM results showing international dispatch are summarized for a no-action case and for the
proposal in Table 5-8 and Table 5-9, respectively. International dispatch only occurred between
Canada and the contiguous United States represented by the IPM regions. Net international
dispatch was also very small as a percentage of total U.S. electricity demand, with electricity
imports less than 1percent for all years and trending towards zero by 2050 for both the no-action
case and proposal.
Table Notes:
* Negative net exports represent imports of electricity
† International dispatch to the contiguous United States only occurred over the U.S. - Canada border.
International dispatch only occurred between Canada and the contiguous United States
represented by the IPM regions. To estimate interregional dispatch, IPM utilizes Total Transfer
Capabilities (TTCs), a metric that represents the capability of the power system to import or
export power reliably from one region to another.
The amount of energy and capacity transferred on a given transmission line between IPM
regions is modeled on a seasonal basis for all run years in the EPA Platform v6. All the modeled
transmission lines have the same TTCs for all seasons. The maximum values for these metrics
were obtained from public sources such as market reports and regional transmission plans,
5-21
wherever available. Where public sources were not available, the maximum values for TTCs are
based on ICF’s expert view. ICF analyzes the operation of the grid under normal and
contingency conditions, using industry-standard methods, and calculates the transfer capabilities
between regions. To calculate the transfer capabilities, ICF uses standard power flow data
developed by the market operators, transmission providers, or utilities, as appropriate. Additional
information regarding power-sector modeling is available via a report submitted to the docket
(U.S. EPA 2023).
5.3 Assessment of PEV Charging Infrastructure
As PEV adoption grows, more charging infrastructure will be needed to support the fleet. This
section summarizes the status and outlook of U.S. PEV charging infrastructure, how much and
what types of charging may be needed to support the level of PEV penetration in the rulemaking,
and how we estimated the associated costs.
5.3.1 Status and Outlook for PEV Charging Infrastructure
5.3.1.1 Definitions
Terminology for charging infrastructure varies in the literature with terms like "charger",
"plug", "outlet", and "port" sometimes being used interchangeably. Throughout this chapter, we
use the following definitions.91 When referring to public charging, a station is the physical
location where charging occurs. Each station may have one or more Electric Vehicle Supply
Equipment (EVSE) ports that provide electricity to a vehicle. The number of vehicles that can
simultaneously charge at the station is equal to the number of EVSE ports. Each port may also
have multiple connectors or plugs, e.g., to accommodate vehicles that use different connector
types, but each port can only charge one vehicle at a time. While it is less common to refer to the
place home charging occurs (e.g., garage or driveway) as a station, we use the term ports in the
same way for residential and non-residential charging.
It must be noted that charging infrastructure is different from the electric power utility
distribution system infrastructure, which is comprised of distribution feeder circuits, switches,
protective equipment, primary circuits, distribution transformers, secondaries, service drops, etc.
The electric power utility distribution system infrastructure typically ends at a service drop (i.e.
the run of cables from the electric power utility's distribution power lines to the point of
connection to a customer's premises).
5.3.1.2 Charging Types
Electric Vehicle Supply Equipment (EVSE) ports can be alternating or direct current (AC or
DC); they also vary by power level. Common AC charging types include L1 (up to about 2 kW
power) and L2 (up to 19.2 kW power) (U.S. Department of Energy, Alternative Fuels Data
Center 2023a) (Schey, Chu and Smart 2022). DC fast charging (DCFC) is available in a range of
power levels today, e.g., 50 kW to 350 kW with standards for even higher-powered DCFC such
as the Megawatt Charging System (MCS) currently in development (CharIN e.V. 2022).
91 Definitions are consistent with those used by (U.S. Department of Energy, Alternative Fuels Data Center 2023a).
A diagram is available at: https://afdc.energy.gov/fuels/electricity_infrastructure.html (last accessed March 8, 2023).
5-22
Generally, the use of higher-power EVSE ports corresponds to faster charging92 though the
maximum power that vehicles can accept varies by model.93
Wireless or inductive charging systems have also been demonstrated and sold as aftermarket
add-ons but have not been widely deployed (U.S. Department of Energy, Alternative Fuels Data
Center 2023a). Due to the uncertainty about the timing and uptake of wireless charging, we
consider it outside the scope of this analysis.
5.3.1.2.1 PEV Charging Infrastructure Status and Trends
Charging infrastructure94 has grown rapidly over the last decade (U.S. Department of Energy,
Alternative Fuels Data Center 2023b). As shown in Figure 5-13, there are more than 50,000 nonresidential
charging stations in the U.S. today with over 140,000 EVSE ports.95 This is an
increase from just over 85,000 EVSE ports as of the end of 2019. These include public EVSE
ports, as well as some private ports, e.g., at workplaces or for fleet use. About 80 percent of
EVSE ports today are L2, however, DCFC deployments have generally experienced faster
growth than L2 in the past few years (Brown, et al. 2022). Among DCFC, there is a trend toward
higher power levels with more than half of the EVSE ports over 50 kW and 10 percent at 300
kW or more as of the first quarter of 2021 (U.S. Department of Energy 2021).
160,000
-
20,000
40,000
60,000
80,000
100,000
120,000
140,000
2011 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022
EVSE Ports Stations
92 For example, DCFC can add 200 miles or more of range per hour of charging compared to about 25 miles for L2,
depending on power levels (U.S. Department of Energy, Alternative Fuels Data Center 2023a).
93 Table 5-1 shows the maximum DCFC power levels we assumed for BEV models in our infrastructure cost
analysis.
94 As used herein, "charging infrastructure" refers to EVSE, which is not a part of electric utility distribution
infrastructure, which is comprised of distribution feeder circuits, switches, protective equipment, primary circuits,
distribution transformers, secondaries, service drops, etc. The electric power utility distribution system infrastructure
typically ends at a service drop (i.e. the run of cables from the electric power utility's distribution power lines to the
point of connection to a customer's premises).
95 These counts may include a small number of EVSE ports and stations at multifamily housing.
5-23
Figure 5-13: U.S. Non-residential PEV Charging Infrastructure from 2011─2022 (Data
Source: (U.S. Department of Energy, Alternative Fuels Data Center 2023b)
While estimates for future infrastructure needs vary widely in the literature, (Brown, et al.
2022) found that the overall ratio of EVSE ports to the number of PEVs on the road today
generally compares favorably to projected needs in national assessments by NREL (E. Wood, et
al. 2017) and ATLAS (McKenzie and Nigro 2021)96. For example, the NREL study estimated a
need for 1.8 DCFC ports for every thousand PEVs on the road, while Atlas estimated the need
for 4.7 DCFC ports per thousand PEVs. By mid-2022, there were 9.2 DCFC ports per thousand
PEVs,97 well above the projected needs estimated by these studies (Brown, et al. 2022). By mid-
2022, there were also 40 public and workplace L2 ports for every thousand PEVs on the road.
This is similar to the 40.1 NREL estimated will be needed, and significantly higher than the 5.8
L2 ports per thousand PEVs that Atlas estimated (Brown, et al. 2022). Of course, keeping up
with charging needs as PEV adoption grows will require continued expansion of, and investment
in, charging infrastructure.
5.3.1.3 PEV Charging Infrastructure Investments
Investments in PEV charging infrastructure have grown rapidly in recent years and are
expected to continue to climb. According to BloombergNEF, annual global investment was $62
billion in 2022, nearly twice that of the prior year, and while about 10 years was needed for
cumulative global investment to total $100 billion, $200 billion could be reached in just three
more years (BloombergNEF 2023). This growth was also seen in U.S. infrastructure spending.
Combined investments in hardware and installation for U.S. home and public charging ports was
over $1.2 billion in 2021, nearly a three-fold increase from 2017 (BloombergNEF 2022).
The U.S. government is making large investments in infrastructure through the Bipartisan
Infrastructure Law (Public Law 117-58 2021) and the Inflation Reduction Act (Public Law 117-
169 2022). However, we expect that private investments will also play a critical role in meeting
future infrastructure needs. Private charging companies have already attracted billions globally
in venture capital and mergers and acquisitions (Hampleton 2023). In the U.S., there was $200
million or more in mergers and acquisition activity in 2022 according to the capital market data
provider PitchBook (St. John and Naughton 2022), indicating strong interest in the future of the
charging industry. Bain projects that by 2030, the U.S. market for electric vehicle charging will
be "large and profitable" with both revenue and profits estimated to grow by a factor of twenty
relative to 2021 (Zayer, et al. 2022).98 Domestic manufacturing capacity is also increasing with
over $600 million in announced investments to support the production of charging equipment
and components at existing or new U.S. facilities. (Joint Office of Energy and Transportation
2023) (Kempower 2023). These activities along with the large variety of private investments
96 NREL and ATLAS both assessed future charging infrastructure needs, but under different PEV adoption
scenarios. See studies for details. Ratios discussed above are based on projected infrastructure needs in 2030.
97 Estimates for the number of DCFC and L2 ports available in 2022 include Tesla EVSE ports that are not currently
available for use by non-Tesla vehicles.
98 Estimates account for hardware and installation as well as operations and other charging services such as vehiclegrid
integration.
5-24
detailed in Chapter 5.3.1.3.4 below suggest that companies are positioning themselves to meet
the growing demand for PEV charging.
The following sections outline some current and upcoming investments in charging
infrastructure from both public and private sources.
5.3.1.3.1 Bipartisan Infrastructure Law
The Bipartisan Infrastructure Law (BIL)99 (Public Law 117-58 2021) provides up to $7.5
billion over five years to build out a national network of PEV chargers. Two-thirds of this
funding is for the National Electric Vehicle Infrastructure (NEVI) Formula Program (U.S.
Department of Transportation, Federal Highway Administration 2022a). The remaining $2.5
billion is for the Charging and Fueling Infrastructure (CFI) Discretionary Grant Program, which
is evenly divided between funds for charging and fueling infrastructure along corridors and in
communities where fueling infrastructure can include hydrogen, propane, or natural gas (U.S.
Department of Transportation, Federal Highway Administration 2022a). These programs are
administered under the Federal Highway Administration with support from the Joint Office of
Energy and Transportation.
The first phase of NEVI formula funding for states was launched in 2022 and is focused on
building out Alternative Fuel Corridors (AFCs) on highways. Charging stations for AFCs are
required to have at least four DCFC ports, each 150 kW or higher (88 FR 12724 2023). Per
FHWA's guidance to states, stations generally must be located no more than 50 miles apart and
one mile from the Interstate (U.S. Department of Transportation, Federal Highway
Administration 2022a). Initial plans for all 50 states, DC, and Puerto Rico covering FY22 and
FY23 funds were approved in September 2022. Together the $1.5 billion in funding will help
deploy or expand charging infrastructure on about 75,000 miles of highway (U.S. Department of
Transportation, Federal Highway Administration 2022b). In March 2023, the first funding
opportunity was opened under the CFI Program with up to $700 million to deploy PEV charging
and hydrogen, propane, or natural gas fueling infrastructure in communities and along corridors
(Joint Office of Energy and Transportation 2023b).
In addition to NEVI, there are a variety of other Federal programs that could help reduce State
or private costs associated with deploying EVSE. For example, constructing and installing
charging infrastructure is an eligible activity for other U.S. Department of Transportation
formula programs including the Congestion Mitigation & Air Quality Improvement Program,
National Highway Performance Program, and Surface Transportation Block Grant Program,
which have a total of more than $40 billion in FY22 funds authorized under the BIL (U.S.
Department of Transportation, Federal Highway Administration 2022a).100 Discretionary grant
programs include the Rural Surface Transportation Grant Program, Infrastructure for Rebuilding
America Grant Program, and the Discretionary Grant Program for Charging and Refueling
Infrastructure (U.S. Department of Transportation, Federal Highway Administration 2022a).
99 Signed into law as the "Infrastructure Investment and Jobs Act"
100 Only a portion is likely to be used to support PEV charging infrastructure, and limits and restrictions may apply.
5-25
5.3.1.3.2 Inflation Reduction Act
The Inflation Reduction Act (IRA), signed into law on August 16, 2022, can also help reduce
the cost that consumers and businesses pay toward PEV charging infrastructure (Public Law
117-169 2022).
Section 13404 extends the Alternative Fuel Refueling Property Tax Credit through Dec 31,
2032, with modifications. Under the new provisions, consumers in low-income or rural areas
would be eligible for a 30 percent credit for the costs of installing a residential charging
equipment subject to a $1,000 cap. Businesses would also be eligible for up to 30 percent of the
costs associated with purchasing and installing charging equipment in these areas (subject to a
$100,000 cap per item) if they meet prevailing wage and apprenticeship requirements. The Joint
Committee on Taxation estimates the cost of this tax credit from FY2022─2031 to be $1.738
billion, which reflects a significant level of support for charging infrastructure and other eligible
alternative fuel property (Joint Committee on Taxation 2022).
5.3.1.3.3 Equity Considerations in BIL and IRA
The infrastructure funding in the BIL and the IRA tax credit discussed above can help to
address equity challenges for PEV charging infrastructure. One of the stated goals of the $7.5
billion in infrastructure funding under the BIL is to support equitable access to charging across
the country (U.S. Department of Transportation, Federal Highway Administration 2022a).
Accordingly, FHWA instructed states to incorporate public engagement in their planning process
for the NEVI Formula program, including reaching out to Tribes, and rural, underserved, and
disadvantaged communities among other stakeholders. This funding will also support the
Justice40 target that 40 percent of the benefits go to disadvantaged communities (U.S.
Department of Transportation, Federal Highway Administration 2022a). Separately,
modifications to the Alternative Fuel Refueling Property Tax Credit in IRA limit applicability to
charging infrastructure installed in low-income or rural census tracts starting in 2023 (Public
Law 117-169 2022). This can help residents in these communities install home charging and
provide an incentive for businesses to site stations in these areas.
5.3.1.3.4 Other Public and Private Investments
States, utilities, auto manufacturers, charging network providers and others are also investing
in and supporting PEV charging infrastructure deployment. California announced plans in 2021
to invest over $300 million in light-duty charging infrastructure and nearly $700 million in
medium- and heavy-duty ZEV infrastructure (California Energy Commission 2021). Several
states including New Jersey and Utah offer partial rebates for residential, workplace, or public
charging while others such as Georgia and D.C. offer tax credits (U.S. Department of Energy,
Alternative Fuels Data Center 2023c).101 The NC Clean Energy Technology Center identified
more than 200 actions taken across 38 states and D.C. related to providing financial incentives
for electric vehicles and or charging infrastructure in 2022, a four-fold increase over the number
of actions in 2017 (Apadula, et al. 2023).102 The Edison Electric Institute estimates that electric
101 Details on eligibility, qualifying expenses, and rebate or tax credit amounts vary by state.
102 Includes actions by states and investor-owned utilities.
5-26
companies have already invested nearly $3.7 billion (EEI 2023).103 And over 60 electric
companies and cooperatives serving customers in 48 states and the District of Columbia have
joined together to advance fast charging through the National Electric Highway Coalition (EEI
2023).
Auto manufacturers are investing in charging infrastructure by offering consumers help with
costs to install home charging or providing support for public charging. For example, GM will
pay for a standard installation of a Level 2 (240 V) outlet for customers purchasing or leasing a
new Bolt (Chevrolet 2023). GM is also partnering with charging provider EVgo to deploy over
2,700 DCFC ports and charging provider FLO to deploy as many as 40,000 L2 ports (GM 2021)
(Joint Office of Energy and Transportation 2023). Volkswagen, Hyundai, and Kia all offer
customers complimentary charging at Electrify America's public charging stations (subject to
time limits or caps) in conjunction with the purchase of select new electric vehicle models (VW
2023) (Hyundai 2023) (Kia 2023). Ford has agreements with several charging providers to make
it easier for their customers to charge and pay across different networks (Ford 2019) and plans to
install publicly accessible DCFC ports at nearly 2,000 dealerships (Joint Office of Energy and
Transportation 2023). Mercedes-Benz recently announced that it is planning to build 2,500
charging points in North America by 2027 (Reuters 2023). Tesla has its own network with over
17,000 DCFC ports and nearly 10,000 L2 ports in the United States (U.S. Department of Energy,
Alternative Fuels Data Center 2023d). Tesla recently announced that by 2024, 7,500 or more
existing and new ports (including 3,500 DCFC) would be open to all PEVs (The White House
2023).
Other charging networks are also expanding. Francis Energy, which has fewer than 1000
EVSE ports today (U.S. Department of Energy, Alternative Fuels Data Center 2023d), aims to
deploy over 50,000 by the end of the decade (Joint Office of Energy and Transportation 2023).
Electrify America plans to more than double its network size (U.S. Department of Energy,
Alternative Fuels Data Center 2023d) to 10,000 fast charging ports across 1800 U.S. and
Canadian stations by 2026. This is supported in part by a $450 million investment from Siemens
and Volkswagen Group (Joint Office of Energy and Transportation 2023). Blink plans to invest
over $60 million to grow its network over the next decade. Charging companies are also
partnering with major retailers, restaurants, and other businesses to make charging available to
customers and the public. For example, EVgo is deploying DCFC at certain Meijer locations,
CBL properties, and Wawa. Volta is installing DCFC and L2 ports at select Giant Food, Kroger,
and Stop and Shop stores, while ChargePoint and Volvo Cars are partnering with Starbucks to
make charging available at select Starbucks locations (Joint Office of Energy and Transportation
2023). Other efforts will expand charging access along major highways at up to 500 Pilot and
Flying J travel centers (through a partnership between Pilot, GM and EVgo) and 200
TravelCenters of America and Petro locations (through a partnership between TravelCenters of
America and Electrify America). BP plans to invest $1 billion toward charging infrastructure by
the end of the decade, including through a partnership to provide charging at various Hertz
locations across the country that could support rental and ridesharing vehicles, taxis, and the
public (Joint Office of Energy and Transportation 2023).
103 The $3.7 billion total includes infrastructure deployments and other customer programs to advance transportation
electrification.
5-27
5.3.2 PEV Charging Infrastructure Cost Analysis
To assess the infrastructure needs and associated costs for this proposal, we start with
estimates of PEV charging demand generated using the methodology described in Chapter 5.1.
The share of demand we anticipate being met by different charging types (e.g., home L2 or
public DCFC) is then used to project the number and mix of EVSE ports that may be needed
each year in the proposal and no-action case. Finally, we assign costs for each EVSE port type
intended to reflect upfront hardware and installation costs based on values in the literature.
We note that the no-action case referred to as part of the infrastructure cost analysis was based
on earlier work with lower projected PEV penetration rates than the no-action case used for
compliance modeling and described in Preamble Section IV.B. (See discussion in DRIA Chapter
5.3.2.6.)
5.3.2.1 Charging Demand Projections
Regionalized PEV charging demand under our proposal was simulated for select years from
2026─2055 under an Interagency Agreement between EPA and the U.S. Department of Energy,
National Renewable Energy Laboratory (NREL). NREL's EVI-X modeling suite was used,
including the EVI-Pro model to simulate charging demand from typical daily travel, EVIRoadTrip
to simulate demand from long-distance travel, and EVI-OnDemand to simulate
demand from ride-hailing applications. Eight unique charging types and locations were
considered: home L1, home L2, work L2, public L2, and public DCFC at 50 kW, 150 kW, 250
kW, and 350 kW power levels (DC-50, DC-150, DC-250, and DC-350). The following
assumptions informed the respective charging shares for daily travel modeled with EVI-Pro:
• PEVs with access to residential charging are assumed to prefer home over either work
or public charging when home charging is sufficient to support all travel needs.
• 75 percent of BEVs and 53 percent of PHEVs are assumed to use L2 for home
charging with the remaining share using L1.104
• Workplace L2 is the next most preferred charging type after home charging.
• Remaining charging needs are met with public charging. DCFC is generally preferred
for BEVs, and among DCFC, the highest power that a vehicle can accept (or "as fast
as possible" charging) is preferred.
• Public L2 charging is used by PHEVs, which are assumed not to be DCFC-capable.
It's also used by BEVs in certain long dwell time location types such as schools or
medical facilities where it's assumed that DCFC is not available.
For road trips and travel by ride-hailing vehicles modeled in EVI-RoadTrip and EVIOnDemand
respectively, all public charging is assumed to be met with DCFC for BEVs.
104 This in part reflects assumptions about the characteristics of PEVs modeled by OMEGA, including a percentage
of low mileage PEVs for which L1 meets daily charging needs.
5-28
Additionally, BEVs able to accept higher-power charging (Gen 2) are assumed to be adopted
more quickly for these applications than for daily travel needs modeled in EVI-Pro.105
As shown in Figure 5-14, the share of PEV charging demand by location and type is similar
for the proposal and no-action case. The majority of PEV charging is home L2 across all years
though the share under the proposal declines from over 70 percent in 2028 to just below 60
percent in 2055 as the share of workplace and public charging grow. DCFC has the next highest
share of demand. Due to the modeling assumption that BEVs charge "as fast as possible" when
using DCFC, 350 kW charging dominates. Since simulated BEV models are capable of higherpower
charging, no DC-50 kW charging is found for either the proposal or no-action case.
Home L1 Home L2 Work L2 Public L2 DC-150 DC-250 DC-350
Figure 5-14: Share of charging demand by location and type for the no-action case (left side
of each pair of bars) and proposal (right side of each pair of bars) for 2028─2055.
5.3.2.2 EVSE Port Counts
The number of EVSE ports needed to meet the level of PEV charging demand in our
proposal and in the no-action case was estimated for all charging types described above. Home
charging was further delineated into charging at single family houses (SFHs)―including both
detached and attached houses (e.g., townhouses) ― and non-SFHs which include apartments,
condos, and mobile homes. Several additional assumptions informed this network sizing. For
home charging, it was assumed that as PEV adoption increases, more home charging ports would
be shared across vehicles. This could reflect SFHs with more than one PEV or residents of multiunit
dwellings that share L2 ports. Specifically, we assume that at 1 percent PEV adoption, 1
EVSE port is needed per PEV with home charging access. This declines to 0.6 EVSE ports per
105 For max DC fast charging rates for different vehicle types modeled in this analysis, see Table 5-1.
5-29
PEV for SFHs and 0.5 EVSE ports per PEV for other home types when PEVs make up the entire
light-duty fleet.
Network sizing for public and workplace charging is based on the regional charging load
profiles described in Chapter 5.1. For each DCFC port type (DC-50, DC-150, DC-250, and DC-
350), the total number of ports needed is scaled such that during the peak hour of usage 20
percent of ports in the region are fully utilized. For work and public L2 charging, 43 percent of
ports are assumed to be fully utilized during the peak hour. These percentages are modeled after
highly utilized stations today (Wood, Borlaug, et al. 2023). 106
Figure 5-15 and Figure 5-16 show the growing charging network that may be needed to meet
PEV charging demand in the proposal and no-action case respectively.107 We anticipate that the
highest number of ports will be needed at homes, growing from under 12 million in 2027 to over
75 million in 2055 under the proposal.108 This is followed by workplace charging, estimated at
about 400,000 EVSE ports in 2027 and over 12.7 million in 2055. Finally, public charging needs
grow from just over 110,000 ports to more than 1.9 million in that timeframe. Notably, while
DCFC at 350 kW constitutes a significant fraction of total electricity demand (Figure 13), the
number of ports needed is relatively small compared to the scale shown. This is because far
fewer 350 kW ports are needed to deliver the same amount of electricity as lower-powered
options. Similar patterns are observed in the no-action case―though fewer total ports are needed
than under the proposal due to the lower anticipated PEV demand.
106 The same method and thresholds for sizing the non-residential charging network based on peak hour of usage
was applied for all years in this analysis. If we instead assumed the percentage of L2 or DCFC ports that are fully
utilized at peak grew as a function of time or PEV penetration, we would expect higher average utilizations per port
and fewer total ports needed.
107 Charging simulations were conducted for 2026, 2028, 2030, 2032, 2035, 2040, 2045, 2050, and 2055. Linear
interpolations were used to estimate the network size in intermediate years. Estimates above do not include PEV
charging demand for medium-duty or heavy-duty vehicles.
108 The number of EVSE ports needed to meet a given level of electricity demand will vary based on the mix of
charging ports, charging preferences, and other factors. Estimates shown reflect assumptions specific to this
analysis, but actual needs could vary.
5-30
EVSE Ports (Millions)
SFH L1 SFH L2 Non-SFH L2 Work L2 Public L2 DC-150 DC-250 DC-350
Figure 5-16: EVSE port counts by charging type for the no-action case 2027─2055.
In order to estimate the costs incurred each year, we calculate how many EVSE ports of each
type would need to be procured and installed to achieve the charging network sizes shown in
5-31
Figure 5-15 and Figure 5-16. There is limited data on the expected lifespan and maintenance
needs of PEV charging infrastructure. We make the simplifying assumption that all EVSE ports
have a 15-year equipment lifetime (Borlaug, Salisbury, et al. 2020). After that, we assume they
must be replaced at full cost. This assumption likely overestimates costs as some EVSE
providers may opt to upgrade existing equipment rather than incur the cost of a full replacement.
Some installation costs such as trenching or electrical upgrades may also not be needed for the
replacement. We do not attempt to estimate EVSE maintenance costs due to uncertainty but note
that maintenance may be able to extend equipment lifetimes. Another simplifying assumption we
make is that EVSE ports are operational and able to meet PEV charging demand the same year
costs are incurred. The actual time to permit and install can vary widely by port type, power
level, region, site conditions and other factors.
5.3.2.3 Hardware & Installation Costs
We assign costs to each of the above infrastructure types intended to reflect the upfront capital
costs associated with procuring and installing the EVSE ports. There are many factors that can
impact equipment costs, including whether ports are wall-mounted or on a pedestal as well as
differences in equipment features and capabilities (Schey, Chu and Smart 2022). For example, an
ICCT paper found that costs more than doubled between networked and non-networked L2
hardware (Nicholas, Estimating electric vehicle charging infrastructure costs across major U.S.
metropolitan areas 2019). Among networked units with one or two ports per pedestal, about a 10
percent difference in per-port hardware costs was found (Nicholas, Estimating electric vehicle
charging infrastructure costs across major U.S. metropolitan areas 2019). The power level of the
EVSE is one of the most significant drivers of cost differences. While estimates for charging
equipment vary across the literature, higher-power charging equipment is typically more
expensive than lower-power units.
Installation costs may include labor, materials (e.g., wire or conduit), permitting, taxes, and
upgrades or modifications to the on-site electrical service. These costs―particularly labor and
permitting―can vary widely by region (Schey, Chu and Smart 2022). They also vary by site. For
example, how much trenching is needed will depend on the distance from where the charging
equipment will be located and the electrical panel. A recent study found that average L2
installation costs at condominiums and commercial locations increased by $16 or $20 for each
extra foot of distance between the EVSE and power source respectively (Schey, Chu and Smart
2022). How many EVSE ports are installed also impacts cost. ICCT estimated that on a per-port
basis, installation costs for 150 kW ports were about 2.5 times higher when only one port is
installed compared to 6─20 per site (Nicholas, Estimating electric vehicle charging infrastructure
costs across major U.S. metropolitan areas 2019). And, as with hardware costs, installation costs
may rise with power levels.
To reflect the diversity of hardware and installation costs, we considered a range of costs for
each charging type as shown in Table 5-10 and detailed below.109
Table 5-14: Cost (hardware and installation) per EVSE port110
Home Work Public
109 All costs shown above and used within the cost analysis are rounded to the nearest hundred.
110 Costs shown are expressed in 2019 dollars, consistent with the original sources from the literature.
5-32
L1 SFH L2 non-SFH L2 L2 L2 DC-50 DC-150 DC-250 DC-350
Low $0 $800 $3,300 $5,100 $5,100 $30,000 $94,000 $124,000 $154,000
Mid $0 $1,100 $3,700 $5,900 $5,900 $56,000 $121,000 $153,000 $185,000
High $0 $1,500 $4,100 $7,300 $7,300 $82,000 $148,000 $182,000 $216,000
5.3.2.3.1 Home Charging Ports
PEVs typically come with a charging cord that can be used for L1 charging by plugging it into
a standard 120 VAC111 outlet, and, in some cases, for L2 charging by plugging into a 240 VAC
outlet.112 We include the cost for this cord as part of the vehicle costs described in Chapter 2, and
therefore don't include it here. We make the simplifying assumption that PEV owners opting for
L1 home charging already have access to a 120 VAC outlet and therefore do not incur
installation costs.113
For L2 home charging, some PEV owners may opt to simply install or upgrade to a 240 VAC
outlet for use with a provided cord while others may choose to purchase or install a wallmounted
or other L2 charging unit, which may have additional features and capabilities. In Table
5-10, the "Low" cost assumes outlet installations only, the "High" cost assumes the purchase and
installation of L2 units, and the "Mid" cost assumes a 50%:50% split.
Costs vary by housing type with installation costs for SFHs typically lower than those for
apartments, condos, or mobile homes (non-SFHs). We use costs by housing type from (Nicholas,
Estimating electric vehicle charging infrastructure costs across major U.S. metropolitan areas
2019) for both outlet upgrades and L2 unit installations.114 For SFH costs, we weight costs for
detached and attached houses by 93 percent to 7 percent.115 We use cost estimates for apartments
to represent all non-SFH home types.
5.3.2.3.2 Work and Public Level 2 Charging Ports
We also source our assumed EVSE costs for work and public AC L2 ports from (Nicholas,
Estimating electric vehicle charging infrastructure costs across major U.S. metropolitan areas
2019).116 We select the lowest per port hardware and installation costs presented for networked
EVSE as our "Low" value and the highest combination of hardware and installation costs
presented as our "High" value. Specifically, we use the following combinations for the costs
shown in Table 5-10:
111 Volts, alternating current.
112 Not all charging cords may be capable of Level 2 charging.
113 (Ge, et al. 2021) found that while residential charging access is expected to decline as PEV adoption grows, the
majority of PEVs are projected to have access to an outlet either where they regularly park or at another parking
location at their home even if PEVs reach 100% of the light-duty fleet.
114 We use costs from Table 5 of (Nicholas 2019), specifically "Level 2 outlet upgrade" for outlet only installations
and "Level 2 charger upgrade" for hardware and installation costs associated with a Level 2 charging unit.
115 Weighting reflects the relative share of light-duty vehicles owned by residents of detached versus attached
houses, sourced from Figure 12 of (Ge, et al. 2021).
116 While (Nicholas 2019) notes that it assumed lower installation costs for workplace charging ports than for public
L2 ports, we make the simplifying assumption that both hardware and installation costs are the same.
5-33
• Low: hardware costs for units with two EVSE ports per pedestal, installation costs for
sites with 6+ EVSE ports outside of California
• Mid: hardware costs for units with two networked EVSE ports per pedestal,
installation costs for sites with 3─5 EVSE ports outside of California
• High: hardware costs for units with one EVSE per pedestal, installation costs for sites
with one EVSE port in California
5.3.2.3.3 Public DC Fast Charging Ports
Cost estimates for DCFC ports are from a 2021 study that drew from various data and
literature sources, including the ICCT report discussed above (Borlaug, Muratori, et al. 2021).
We use the lower end of the ranges presented for procurement and installation costs as the "Low"
costs for 50 kW, 150 kW, and 350 kW DCFCs in Table 5-10, and the upper end of the ranges for
the "High" costs. Our "Mid" costs are the average of "Low" and "High". Since no estimate is
provided for 250 kW DCFCs, we take the average of costs for 150 kW and 350 kW DCFCs.117
5.3.2.4 Will Costs Change Over Time?
The infrastructure costs shown above reflect present day costs (expressed in 2019 dollars).
However, both hardware and installation costs could vary over time. For example, hardware
costs could decrease due to manufacturing learning and economies of scale. Recent studies by
ICCT assumed a 3 percent annual reduction in hardware costs (Nicholas, Estimating electric
vehicle charging infrastructure costs across major U.S. metropolitan areas 2019) (Bauer, Hsu, et
al., Charging Up America: Assessing the Growing Need for U.S. Charging Infrastructure
Through 2030 2021). By contrast, installation costs could increase due to growth in labor or
material costs. As noted above, installation costs also depend on site conditions, including
whether sufficient electric capacity exists to add charging infrastructure and how much trenching
is required between the EVSE port and electrical panel. If easier and, therefore, lower cost sites
are selected first, then over time installation costs could rise as charging stations start to be
installed in more challenging locations. (Bauer, Hsu, et al., Charging Up America: Assessing the
Growing Need for U.S. Charging Infrastructure Through 2030 2021) found that these and other
countervailing factors could result in the average cost of a 150 kW EVSE port in 2030 being
similar (~3 percent lower) to that in 2021.
Due to the uncertainty on how costs may change over time, we have made the simplifying
assumption for this analysis to keep combined hardware and installation costs per EVSE port
constant.
5.3.2.5 Other Considerations
EPA acknowledges that there may be additional infrastructure needs and costs beyond those
associated with charging equipment itself. While planning for additional electricity demand is a
standard practice for utilities and not specific to PEV charging, the buildout of public and private
charging stations (particularly those with multiple high-powered DC fast charging units) could in
117 Costs may not scale linearly with power level. We take the average as a simplifying assumption but continue to
monitor the literature for costs associated with this power level.
5-34
some cases require upgrades to local distribution systems. For example, a recent study found
power needs as low as 200 kW could trigger the need to install a distribution transformer while a
load of 5 MW or more could require upgrades to feeder circuits or the addition of a feeder
breaker (Borlaug, Muratori, et al. 2021).
There are a variety of approaches that could reduce the need or scale of such upgrades—
potentially saving both cost and deployment time. For example, distribution system capacity and
interconnection could be factored into the site selection process, and when possible, utilities
could work with station developers to evaluate multiple potential sites before a selection is made
(Hernandez 2022). Another emerging best practice identified by the Interstate Renewable Energy
Council is for utilities to provide hosting capacity maps (HCMs) that identify grid capacity
constraints (Hernandez 2022). Such maps could help developers determine whether area feeders
or substations have additional capacity for charging or other loads. By mid-2022, requirements
for HCMs or related analyses were in place in ten states identified by Lawrence Berkeley
National Laboratory (Schwartz 2022). More broadly, 25 states and the District of Columbia have
ongoing efforts and requirements to support proactive distribution system planning and grid
modernization (Schwartz 2022).
Managing the additional demand from PEV charging is another key strategy. Automated load
management or power control systems are being explored as a way to dynamically limit total
charging load and ensure it doesn't exceed available capacity―potentially reducing the need for
upgrades at some sites (Nuvve and Enel X 2020) (BATRIES 2023). The use of onsite battery
storage and renewables may also be able to reduce demand on the grid, and some station
operators may opt for these technologies to mitigate demand charges associated with peak power
(Alexander, et al. 2021). In addition, managed or smart charging can be used in some cases to
reduce power or shift charging demand to times when it is easier to meet. Charging equipment
funded under the NEVI Formula Program, or as part of publicly-accessible charging projects
funded under Title 23, U.S.C., must be capable of smart charge management (88 FR 12724
2023).118 Finally, we note that an adapter developed by Argonne National Laboratory to retrofit
non-networked L2 EVSE to allow load management and other smart charging capabilities is in
the process of being commercialized (EVmatch, Inc. 2023). (Also see the discussion of managed
charging and vehicle-grid integration in Chapter 5.4 below.)
Innovative charging approaches may also reduce the need for upgrades in certain cases, or
otherwise reduce infrastructure costs. Mobile charging units could be a solution for locations like
parking garage decks in which it is challenging or costly to install EVSE ports (Alexander, et al.
2021), or be used as a temporary solution while stations are being built. These units are available
in a variety of power levels (e.g., the dual-port Mobi EV charger by (FreeWire Technologies
2023) can provide up to 11 kW while the Lightning Mobile unit can be configured to have up to
five 80 kW DCFC ports (Lightning eMotors 2023)), and can be recharged at times and locations
in which there is sufficient electrical capacity. Standalone charging canopies with integrated
118 The National Electric Vehicle Standards and Requirements Final Rule establishes requirements for standardized
communication among vehicles, charging equipment, and networks to ensure interoperability. Specifically, the use
of ISO 15118 is required for communication between vehicles and chargers, Open Charge Point Protocol for
communication between chargers and networks, and Open Charge Point Interface for communication among
charging networks. (See (88 FR 12724 2023) for details on applicable versions and the timing for these
requirements.)
5-35
solar cells and battery storage that don't need to be connected to the grid (Alexander, et al. 2021)
may be useful for remote locations or where construction is costly or difficult.
There is considerable uncertainty associated with future distribution upgrade needs as well as
with the uptake of the technologies and approaches discussed above that could reduce upgrade
costs, and we do not model them directly as part of our infrastructure cost analysis.119
5.3.2.6 PEV Charging Infrastructure Cost Summary
Table 5-11 shows the estimated annual PEV charging infrastructure costs for the indicated
calendar years in the proposal relative to the no action case using the "Low", "Mid", and "High"
per port cost estimates discussed above.120 Annual costs range from $0.6 billion dollars under the
low scenario to $10 billion under the high scenario. The table also shows the present value (PV)
of these costs and the equivalent annualized value (EAV) for the calendar years 2027–2055 using
both 3 percent and 7 percent discount rates. The "Mid" costs are included as social costs in the
net benefits estimates for this proposal, presented in Chapter 10.6.
As previously noted, the no-action case used throughout the PEV charging infrastructure cost
analysis was based on earlier work with lower projected PEV penetration rates than the no-action
case used for compliance modeling. As a result, the number of EVSE ports and associated costs
for the no-action scenario discussed in this section are likely lower than they would be under the
compliance no-action case. Since we estimate costs for the proposal relative to the no-action
119 The per port EVSE costs shown in Table 5-10 may include some distribution system costs. For example,
(Nicholas 2019) notes that public and workplace installation costs include "utility upgrades". We don't add to, or
otherwise adjust, these values to account for transformer upgrades or other potential upstream distribution costs
specific to the projected port counts in this analysis.
120 See spreadsheet "PEV Charging Infrastructure Cost Analysis" in the docket.
5-36
case, the resulting EVSE costs shown in Table 5-11 are likely to be conservative, or higher, than
if we had applied the same no-action case used for compliance modeling.
5.4 Grid Resiliency
How the additional electricity demand from PEVs will impact the grid will depend on many
factors including the time-of-day that charging occurs, and the use of battery storage and vehicleto-
grid (V2G) or other Vehicle-Grid Integration (VGI) technology. For example, PEVs can be
scheduled to charge at off-peak hours when the electricity demand is easier to meet. Onsite
battery storage, if deployed at charging stations, could also reduce potential grid impacts by
shifting when electricity is drawn from the grid while still providing power to vehicles when
needed. Managed charging and battery storage could also enable increasing renewable use if
charging load is shifted to times with excess solar or wind that might otherwise be curtailed.
V2G technology, which allows electricity to be drawn from vehicles when not in use, could even
allow PEVs to enhance grid reliability.
Electric power system reliability can be determined using a variety of statistical metrics. The
generally accepted metrics by which electric utilities across the U.S. measure and report electric
power system reliability is set by the Institute of Electrical and Electronics Engineers (IEEE)
using the standard IEEE 1366-2022 (IEEE Guide for Electric Power Distribution Reliability
Indices). The formulation of overall electric power system reliability metrics includes electric
power outages associated with what is known as “loss of supply” events; these are events in
which electric power generation and/or electric power transmission is the root cause for a power
outage. As this discussion is limited to electric power distribution system reliability, an electric
power system reliability metric that excludes electric power outages associated with the loss of
supply events (i.e. loss of electric power generation and/or electric power transmission) is
appropriate.
Using this approach, we observed that electric power utilities in 48 U.S. Census Division and
State regions tracked by the U.S. Energy Information Administration (EIA) had overall trends in
distribution grid reliability that were less than the national average for the years 2013 and 2021
(the most-recent years for which data is available) (EIA, 2022). Conversely, 13 U.S. Census
Division and State regions had overall trends in distribution grid reliability for the same years
that were greater than the national average for the years in question. According to the California
Public Utilities Commission, "This data alone does not fully capture the current state of
reliability of the U.S. electric power distribution system…" (Enis 2021). Given the massive size
of the electric power distribution system – with its multitude of regional, climate, and density
variations – interpreting distribution system reliability indices can be challenging to interpret.
Moreover, such reliability statistics focus on outage duration and customer counts, which may
obscure important regional variations. However, as the expected increase in electricity
generation associated with the proposal relative to a no action case is relatively small –
approximately 4.4 percent increase in 2050 – we do not expect the U.S. electric power
distribution system to be adversely affected by the projected additional number of charging
electric vehicles.
Grid reliability is not expected to be adversely affected by the modest increase in electricity
demand associated with electric vehicle charging. As shown in Figure 5-8, we project the
additional generation needed to meet the demand of PEVs in the proposal to be relatively modest
compared to the no-action case, ranging from less than 0.4percent in 2030 to approximately
5-37
4.4percent in 2050. The California Public Utilities Commission (CPUC) (California Public
Utilities Commission 2022) and the California Energy Commission (CEC) (Lipman, Harrington
and Langton 2021) (Chhaya, et al. 2019) have been actively engaged in VGI121 efforts for over a
decade, along with the California Independent System Operator (CAISO) (California
Independent System Operator 2014, California Energy Commission; California Public Utilities
Commission; Governor's Office of Business and Economic Development 2021), large private
and public electrical utilities (SCE, PG&E, SDG&E, etc.), several automakers (Ford, GM, FCA,
BMW, Audi, Nissan, Toyota, Honda, and others), and EV charger companies, the Electric Power
Research Institute (EPRI), and various other research organizations.
These efforts (Lipman, Harrington and Langton 2021) demonstrated the ability to shift up to
20 percent of electric vehicle charging loads in any given hour to other times of the day as well
as the ability to add up to 30 percent of electric vehicle charging loads in a given hour (Lipman,
Harrington and Langton 2021). We anticipate similar strategies could be used to shift PEV
charging loads from peak times as needed to reduce grid impacts across different regions. As the
expected increase in electric power demand resulting from PEV charging in this proposal will be
well-under 20 percent, we do not anticipate it to pose grid reliability issues.
The increasing integration of electric vehicle charging into the electric power grid has also
been found to increase grid reliability (Chhaya, et al. 2019) , as the ability to shift and curtail
electric power loads improves grid operations and, therefore, grid reliability. Such integration
has been found to create value for electric vehicle drivers, electric grid operators, and ratepayers.
Management of PEV charging can reduce overall costs to utility ratepayers by delaying electric
utility customer rate increases associated with equipment upgrades and may allow utilities to use
electric vehicle charging as a resource to manage intermittent renewables or provide ancillary
services.
The Electric Power Research Institute (EPRI)122, is undertaking a three year-long research
project to better-understand the scale of commitment and investment in the electric power grid
that is required to meet the anticipated electric power loads. Thus far, the electric power sector
and its regulators have focused on incremental EV load growth and charger utilization (Electric
Power Research Institute 2022). The work of EPRI focuses on grid impacts and associated lead
times required to better-prepare the grid (including transmission, substation, feeder, and
transformer) for vehicle electrification. These efforts are, in part, based upon grid reliability
research conducted by EPRI (Maitra 2013) (Electric Power Research Institute 2012), which
identified grid and charging behavior characteristics associated with grid resiliency. We also
consulted with FERC staff on distribution system reliability and related issues.
State government plays an important role in vehicle electrification (including aspects of grid
resilience), as most electric utilities are regulated by state Public Service Commissions (PSC)
and Public Utility Commissioners (PUC) and since Federal funding for vehicle electrification is
largely distributed through state agencies. The National Association of Regulatory Utility
Commissioners (NARUC), a national association representing the state public service
commissioners who regulate essential utility services, including energy, telecommunications, and
121 VGI is also sometimes referred to as Vehicle-to-Grid or VTG or V2G.
122 EPRI is an independent, nonprofit, U.S.-based organization that conducts research and development related to the
generation, delivery, and use of electricity [https://www.epri.com/].
5-38
water, produced a series of documents aimed at providing vehicle electrification-related guidance
for state regulators (National Association of Regulatory Utility Commissioners 2022a),
facilitating electric vehicle interoperability (National Association of Regulatory Utility
Commisioners 2022b), and fostering vehicle electrification equity (National Association of
Regulatory Utility Commissioners 2022c). NARUC, in conjunction with the National
Association of State Energy Officials (NASEO) and the American Association of State Highway
and Transportation Officials (AASHTO), also produced a guide for public utility commissions,
state energy offices, and departments of transportation discussing the state-level roles and their
interrelations vis-à-vis transportation electrification (National Council on Electricity Policy,
National Association of Regulatory Utility Commissioners 2022).
We also note that DOE is engaged in multiple efforts to modernize the grid and improve
resilience and reliability. For example, in November 2022, DOE announced $13 billion in
funding opportunities under BIL to support transmission and distribution infrastructure. This
includes $3 billion for smart grid grants with a focus on PEV integration among other topics
(U.S. Department of Energy 2022).
5-39

"""

# Process the PDF content
output_json_path = "chapter 5_structure.json"

structure = parse_pdf_structure_with_page_numbers(pdf_text, output_json_path)

# Output the structure as a JSON-like dictionary

#print(json.dumps(structure, indent=4))


 
# 
# 
# # Example usage
# pdf_path = "chapter 4.pdf"
# 
# parse_pdf_to_json(pdf_path, output_json_path)
# 

# import re
# 
# def parse_pdf_structure(pdf_text):
#     # Initialize structure for chapters
#     chapters = []
#     current_chapter = None
#     current_section = None
#     current_subsection = None
# 
#     # Split the text into lines for processing
#     lines = pdf_text.splitlines()
# 
#     for line in lines:
#         # Check for chapter titles (e.g., "Chapter 4: Consumer Impacts")
#         chapter_match = re.match(r'^(Chapter \d+:\s.*)', line)
#         if chapter_match:
#             if current_chapter:
#                 chapters.append(current_chapter)
#             current_chapter = {
#                 "chapter_title": chapter_match.group(1),
#                 "sections": []
#             }
#             current_section = None  # Reset the section for each new chapter
#             continue
#         
#         # Check for section titles (e.g., "4.1 Modeling the Purchase Decision")
#         section_match = re.match(r'^\d+\.\d+(?:\.\d+)*\s+(.*)', line)
#         if section_match:
#             if current_section:
#                 current_chapter["sections"].append(current_section)
#             current_section = {
#                 "title": section_match.group(1),
#                 "subsections": [],
#                 "content": ""
#             }
#             continue
#         
#         # Check for subsection titles (e.g., "4.1.1 Costs Incorporated in the Purchase Decision")
#         subsection_match = re.match(r'^\d+\.\d+\.\d+\s+(.*)', line)
#         if subsection_match:
#             if current_subsection:
#                 current_section["subsections"].append(current_subsection)
#             current_subsection = {
#                 "title": subsection_match.group(1),
#                 "content": ""
#             }
#             continue
#         
#         # Append content to the current section or subsection
#         if current_subsection:
#             current_subsection["content"] += " " + line.strip()
#         elif current_section:
#             current_section["content"] += " " + line.strip()
# 
#     # Add the last section/subsection if present
#     if current_subsection:
#         current_section["subsections"].append(current_subsection)
#     if current_section:
#         current_chapter["sections"].append(current_section)
#     if current_chapter:
#         chapters.append(current_chapter)
# 
#     return {"chapters": chapters}
# 
# # Example PDF text extraction (as a string)
# pdf_text = """Chapter 4: Consumer Impacts and Related Economic Considerations
# This chapter discusses the impacts of the proposed rule on consumers and related economic...
# 4.1 Modeling the Purchase Decision
# In this section, we focus our discussion on our modeling of the consumer purchase decision...
# 4.1.1 Costs Incorporated in the Purchase Decision
# During the vehicle purchase decision process, consumers reference a wide variety of information...
# 4.1.2 Consumer Response to Costs and Perceptions of Technology
# Total sales are determined as described in Chapter 4.4 below...
# 4.2 Ownership Experience
# Having described how we model the consumer purchase decision in Chapter 4.1...
# 4.2.1 Vehicle Miles Traveled and Rebound Effect
# Critical to estimating the impacts of emissions standards is the number of vehicle miles traveled...
# """
# 
# # Process the PDF content
# structure = parse_pdf_structure(pdf_text)
# 
# # Output the structure as a JSON-like dictionary
# import json
# print(json.dumps(structure, indent=4))
#-------------------------------
# import fitz  # PyMuPDF
# import re
# import json
# 
# def parse_pdf_to_json(pdf_path, output_json_path):
#     # Open the PDF file
#     document = fitz.open(pdf_path)
#     
#     # Initialize JSON structure
#     json_data = {"chapters": []}
# 
#     current_chapter = None
#     current_section = None
#     current_subsection = None
# 
#     for page_num in range(len(document)):
#         page = document[page_num]
#         text = page.get_text()
#         lines = text.splitlines()
# 
#         for line in lines:
#             # Match chapter titles (e.g., "4.2 Ownership Experience")
#             chapter_match = re.match(r'^(\d+\.\d+)\s+(.*)', line)
#             if chapter_match:
#                 chapter_number, chapter_title = chapter_match.groups()
#                 if current_chapter:
#                     json_data["chapters"].append(current_chapter)
#                 current_chapter = {
#                     "chapter_title": chapter_title.strip(),
#                     "sections": []
#                 }
#                 current_section = None
#                 current_subsection = None
#                 continue
# 
#             # Match section titles (e.g., "4.2.1 Vehicle Miles Traveled and Rebound Effect")
#             section_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.*)', line)
#             if section_match:
#                 section_number, section_title = section_match.groups()
#                 if current_section:
#                     current_chapter["sections"].append(current_section)
#                 current_section = {
#                     "section_number": section_number.strip(),
#                     "section_title": section_title.strip(),
#                     "subsections": [],
#                     "content": "",
#                     "page_number": page_num + 1
#                 }
#                 current_subsection = None
#                 continue
# 
#             # Match subsection titles (optional further hierarchy)
#             subsection_match = re.match(r'^(\d+\.\d+\.\d+\.\d+)\s+(.*)', line)
#             if subsection_match:
#                 subsection_number, subsection_title = subsection_match.groups()
#                 if current_subsection:
#                     current_section["subsections"].append(current_subsection)
#                 current_subsection = {
#                     "subsection_number": subsection_number.strip(),
#                     "subsection_title": subsection_title.strip(),
#                     "content": "",
#                     "page_number": page_num + 1
#                 }
#                 continue
# 
#             # Add content to the appropriate level
#             content_line = line.strip()
#             if current_subsection:
#                 current_subsection["content"] += content_line + " "
#             elif current_section:
#                 current_section["content"] += content_line + " "
#         
#         if current_subsection:
#                 # Ensure current_section is valid and properly structured
#             if current_section is not None and "subsections" in current_section:
#                 # Append the current_subsection only if it exists
#                 if current_subsection is not None:
#                     current_section["subsections"].append(current_subsection)
#                     current_subsection = None
#             else:
#                 # Handle cases where current_section is not properly initialized
#                 print(f"Error: current_section is not initialized or missing 'subsections'. Skipping line or fixing structure.")
#                 # Optionally initialize a default current_section to avoid further issues
#                 if current_section is None:
#                     current_section = {
#                         "title": "Untitled Section",
#                         "subsections": [],
#                         "content": ""
#                     }
#                 elif "subsections" not in current_section:
#                     current_section["subsections"] = []
# 
#                 # Append the current_subsection to the newly initialized or fixed section
#                 if current_subsection is not None:
#                     current_section["subsections"].append(current_subsection)
#                     current_subsection = None
# 
#            
#         # Ensure current_section is not None before appending
#         if current_section:
#         # Append the completed current_subsection to the current_section
#             if current_subsection:
#                 current_section["subsections"].append(current_subsection)
#                 current_subsection = None
#         else:
#             # Handle case where current_section is None (unexpected structure)
#             print(f"Error: Attempted to add subsection to a non-existent section. Line: {line}")
#             # Optionally initialize a placeholder to prevent NoneType errors
#             current_section = {
#                 "title": "Unknown Section",
#                 "subsections": [],
#                 "content": ""}
# 
#         # After processing all lines, append the last subsection/section/chapter
#        
# #         if current_section:
# #             current_chapter["sections"].append(current_section)
# #             current_section = None
#     if current_chapter:
#         json_data["chapters"].append(current_chapter)
# 
#     # Save the JSON structure to the output file
#     with open(output_json_path, 'w', encoding='utf-8') as json_file:
#         json.dump(json_data, json_file, indent=4)
# 
#     print(f"JSON file saved to {output_json_path}")
# 
# 
# # Example usage
# pdf_path = "chapter 4.pdf"
# output_json_path = "chapter 4.json"
# parse_pdf_to_json(pdf_path, output_json_path)
# 