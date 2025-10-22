/**
 * Character Name Randomizer - Privacy Enhancement Module
 *
 * This module provides anonymization functionality for user displays by replacing
 * actual user names with randomly assigned character names from popular culture.
 *
 * Key Features:
 * - 721+ unique character names from movies, TV shows, comics, and more
 * - Session-based randomization (names change on each page reload)
 * - Guaranteed uniqueness within a session (no duplicate pseudonyms)
 * - Automatic overflow handling with Roman numeral suffixes (e.g., "Iron Man II")
 * - Zero impact on backend database (frontend display layer only)
 *
 * Privacy Benefits:
 * - Protects user identity when sharing screenshots or demos
 * - Enables safe data analysis without exposing real user names
 * - Useful for presentations, documentation, and training materials
 *
 * Usage:
 * Call getRandomCharacterName(userId) to get a unique character name for any user ID.
 * The same user ID will always return the same character name within a session,
 * but will get a new random name when the page is refreshed.
 *
 * @module character-names
 */

/**
 * Master list of character names used for pseudonym generation.
 * This array contains 721+ names from various sources including:
 * - Superheroes (Marvel, DC)
 * - Cartoons (SpongeBob, Simpsons, Rick & Morty, etc.)
 * - Sci-Fi (Star Wars, Star Trek, Doctor Who, etc.)
 * - Fantasy (Harry Potter, Lord of the Rings, Game of Thrones)
 * - Sitcoms (Friends, The Office, Parks & Rec, etc.)
 * - Drama (The Sopranos, The Wire, Breaking Bad, etc.)
 * - Comedy (30 Rock, Arrested Development, etc.)
 * - Scientists (Einstein, Curie, Hawking, etc.)
 *
 * @constant
 * @type {string[]}
 */
const CHARACTER_NAMES = [
  // Superheroes
  "Iron Man",
  "Captain America",
  "Black Widow",
  "Thor",
  "Hulk",
  "Spider-Man",
  "Black Panther",
  "Doctor Strange",
  "Scarlet Witch",
  "Vision",
  "Ant-Man",
  "Wasp",
  "Hawkeye",
  "Falcon",
  "Winter Soldier",
  "Star-Lord",
  "Gamora",
  "Rocket Raccoon",
  "Groot",
  "Drax",
  "Superman",
  "Batman",
  "Wonder Woman",
  "The Flash",
  "Green Lantern",
  "Aquaman",
  "Cyborg",
  "Green Arrow",
  "Supergirl",
  "Shazam",

  // Cartoon Characters
  "SpongeBob",
  "Patrick Star",
  "Squidward",
  "Sandy Cheeks",
  "Bugs Bunny",
  "Daffy Duck",
  "Porky Pig",
  "Tweety Bird",
  "Scooby-Doo",
  "Shaggy",
  "Fred Jones",
  "Daphne Blake",
  "Velma Dinkley",
  "Homer Simpson",
  "Marge Simpson",
  "Bart Simpson",
  "Lisa Simpson",
  "Peter Griffin",
  "Stewie Griffin",
  "Brian Griffin",
  "Lois Griffin",
  "Meg Griffin",
  "Eric Cartman",
  "Stan Marsh",
  "Kyle Broflovski",
  "Kenny McCormick",
  "Rick Sanchez",
  "Morty Smith",
  "Summer Smith",
  "Jerry Smith",
  "Beth Smith",
  "Mickey Mouse",
  "Donald Duck",
  "Goofy",
  "Minnie Mouse",
  "Pluto",
  "Tom Cat",
  "Jerry Mouse",
  "Popeye",
  "Olive Oyl",
  "Bluto",
  "Charlie Brown",
  "Snoopy",
  "Lucy van Pelt",
  "Linus van Pelt",
  "Fred Flintstone",
  "Barney Rubble",
  "Wilma Flintstone",
  "Betty Rubble",
  "George Jetson",
  "Jane Jetson",
  "Elroy Jetson",
  "Judy Jetson",
  "Astro",
  "Rosie the Robot",
  "Yogi Bear",
  "Boo-Boo Bear",
  "Huckleberry Hound",
  "Top Cat",
  "Pink Panther",

  // Anime/Manga Characters
  "Goku",
  "Vegeta",
  "Gohan",
  "Piccolo",
  "Naruto",
  "Sasuke",
  "Sakura",
  "Kakashi",
  "Luffy",
  "Zoro",
  "Nami",
  "Sanji",
  "Chopper",
  "Pikachu",
  "Ash Ketchum",
  "Misty",
  "Brock",
  "Sailor Moon",
  "Tuxedo Unsure",
  "Edward Elric",
  "Alphonse Elric",
  "Spike Spiegel",

  // Star Wars
  "Luke Skywalker",
  "Leia Organa",
  "Han Solo",
  "Darth Vader",
  "Obi-Wan Kenobi",
  "Yoda",
  "Chewbacca",
  "R2-D2",
  "C-3PO",
  "Rey",
  "Finn",
  "Poe Dameron",
  "Kylo Ren",
  "Ahsoka Tano",
  "Mando",
  "Baby Yoda",
  "Boba Fett",

  // Star Trek
  "James Kirk",
  "Spock",
  "Leonard McCoy",
  "Jean-Luc Picard",
  "William Riker",
  "Data",
  "Worf",
  "Geordi La Forge",
  "Beverly Crusher",
  "Deanna Troi",

  // Harry Potter
  "Harry Potter",
  "Hermione Granger",
  "Ron Weasley",
  "Albus Dumbledore",
  "Severus Snape",
  "Hagrid",
  "Draco Malfoy",
  "Luna Lovegood",
  "Neville Longbottom",
  "Sirius Black",

  // Lord of the Rings
  "Frodo Baggins",
  "Gandalf",
  "Aragorn",
  "Legolas",
  "Gimli",
  "Sam Gamgee",
  "Merry Brandybuck",
  "Pippin Took",
  "Boromir",
  "Galadriel",

  // Game of Thrones
  "Jon Snow",
  "Daenerys Targaryen",
  "Tyrion Lannister",
  "Arya Stark",
  "Sansa Stark",
  "Cersei Lannister",
  "Jaime Lannister",
  "Ned Stark",

  // The Office
  "Michael Scott",
  "Jim Halpert",
  "Pam Beesly",
  "Dwight Schrute",
  "Ryan Howard",
  "Kelly Kapoor",
  "Andy Bernard",
  "Angela Martin",
  "Kevin Malone",
  "Oscar Martinez",

  // Friends
  "Ross Geller",
  "Rachel Green",
  "Monica Geller",
  "Chandler Bing",
  "Joey Tribbiani",
  "Phoebe Buffay",

  // Breaking Bad
  "Walter White",
  "Jesse Pinkman",
  "Skyler White",
  "Hank Schrader",
  "Saul Goodman",
  "Mike Ehrmantraut",

  // The Matrix
  "Neo",
  "Morpheus",
  "Trinity",
  "Agent Smith",

  // Back to the Future
  "Marty McFly",
  "Doc Brown",
  "Biff Tannen",

  // Jurassic Park
  "Alan Grant",
  "Ellie Sattler",
  "Ian Malcolm",
  "John Hammond",

  // Indiana Jones
  "Indiana Jones",
  "Marion Ravenwood",
  "Short Round",

  // James Bond
  "James Bond",
  "M",
  "Q",
  "Moneypenny",

  // Ghostbusters
  "Peter Venkman",
  "Ray Stantz",
  "Egon Spengler",
  "Winston Zeddemore",

  // Terminator
  "Sarah Connor",
  "Kyle Reese",
  "T-800",
  "John Connor",

  // Alien
  "Ellen Ripley",

  // Die Hard
  "John McClane",

  // Rocky
  "Rocky Balboa",
  "Apollo Creed",
  "Mickey Goldmill",

  // Karate Kid
  "Daniel LaRusso",
  "Mr. Miyagi",
  "Johnny Lawrence",

  // Forrest Gump
  "Forrest Gump",
  "Jenny Curran",
  "Lieutenant Dan",

  // The Godfather
  "Vito Corleone",
  "Michael Corleone",
  "Sonny Corleone",
  "Fredo Corleone",

  // Toy Story
  "Woody",
  "Buzz Lightyear",
  "Jessie",
  "Rex",
  "Hamm",
  "Slinky Dog",
  "Mr. Potato Head",
  "Mrs. Potato Head",

  // Finding Nemo
  "Nemo",
  "Marlin",
  "Dory",
  "Crush",
  "Squirt",

  // The Incredibles
  "Mr. Incredible",
  "Elastigirl",
  "Violet Parr",
  "Dash Parr",
  "Jack-Jack",
  "Frozone",

  // Monsters Inc
  "Sulley",
  "Mike Wazowski",
  "Boo",
  "Randall",

  // Shrek
  "Shrek",
  "Donkey",
  "Fiona",
  "Puss in Boots",

  // Frozen
  "Elsa",
  "Anna",
  "Olaf",
  "Kristoff",
  "Sven",

  // The Lion King
  "Simba",
  "Mufasa",
  "Nala",
  "Timon",
  "Pumbaa",
  "Scar",
  "Rafiki",
  "Zazu",

  // Aladdin
  "Aladdin",
  "Jasmine",
  "Genie",
  "Jafar",
  "Abu",
  "Iago",

  // Beauty and the Beast
  "Belle",
  "Beast",
  "Gaston",
  "Lumiere",
  "Cogsworth",
  "Mrs. Potts",
  "Chip",

  // The Little Mermaid
  "Ariel",
  "Eric",
  "Sebastian",
  "Flounder",
  "Ursula",
  "King Triton",

  // Mulan
  "Mulan",
  "Mushu",
  "Li Shang",

  // Tangled
  "Rapunzel",
  "Flynn Rider",
  "Pascal",
  "Maximus",

  // Moana
  "Moana",
  "Maui",
  "Heihei",
  "Pua",

  // Encanto
  "Mirabel",
  "Isabella",
  "Luisa",
  "Bruno",
  "Abuela Alma",

  // Turning Red
  "Mei Lee",
  "Ming Lee",

  // Inside Out
  "Joy",
  "Sadness",
  "Anger",
  "Fear",
  "Disgust",

  // WALL-E
  "WALL-E",
  "EVE",

  // Up
  "Carl Fredricksen",
  "Russell",
  "Dug",
  "Kevin",

  // Ratatouille
  "Remy",
  "Linguini",
  "Colette",

  // Cars
  "Lightning McQueen",
  "Mater",
  "Sally Carrera",

  // Avatar: The Last Airbender
  "Aang",
  "Katara",
  "Sokka",
  "Toph",
  "Zuko",
  "Iroh",
  "Azula",

  // Stranger Things
  "Eleven",
  "Mike Wheeler",
  "Dustin Henderson",
  "Lucas Sinclair",
  "Will Byers",
  "Joyce Byers",
  "Jim Hopper",
  "Steve Harrington",
  "Nancy Wheeler",
  "Jonathan Byers",
  "Max Mayfield",
  "Eddie Munson",

  // The Mandalorian
  "Din Djarin",
  "Grogu",
  "Cara Dune",
  "Greef Karga",

  // The Witcher
  "Geralt of Rivia",
  "Yennefer",
  "Ciri",
  "Jaskier",

  // Sherlock
  "Sherlock Holmes",
  "John Watson",
  "Mycroft Holmes",
  "Jim Moriarty",

  // Doctor Who
  "The Doctor",
  "Rose Tyler",
  "Amy Pond",
  "Rory Williams",
  "Clara Oswald",
  "River Song",

  // Supernatural
  "Dean Winchester",
  "Sam Winchester",
  "Castiel",
  "Bobby Singer",

  // The Walking Dead
  "Rick Grimes",
  "Daryl Dixon",
  "Michonne",
  "Carol Peletier",
  "Glenn Rhee",
  "Maggie Greene",
  "Negan",

  // Ted Lasso
  "Ted Lasso",
  "Roy Kent",
  "Keeley Jones",
  "Rebecca Welton",
  "Coach Beard",

  // Parks and Recreation
  "Leslie Knope",
  "Ron Swanson",
  "Tom Haverford",
  "April Ludgate",
  "Andy Dwyer",
  "Ben Wyatt",

  // Brooklyn Nine-Nine
  "Jake Peralta",
  "Amy Santiago",
  "Rosa Diaz",
  "Charles Boyle",
  "Terry Jeffords",
  "Raymond Holt",
  "Gina Linetti",

  // Community
  "Jeff Winger",
  "Annie Edison",
  "Abed Nadir",
  "Troy Barnes",
  "Britta Perry",
  "Shirley Bennett",
  "Pierce Hawthorne",

  // How I Met Your Mother
  "Ted Mosby",
  "Marshall Eriksen",
  "Lily Aldrin",
  "Barney Stinson",
  "Robin Scherbatsky",

  // The Sopranos
  "Tony Soprano",
  "Carmela Soprano",
  "Christopher Moltisanti",
  "Paulie Walnuts",
  "Silvio Dante",
  "Dr. Jennifer Melfi",
  "Meadow Soprano",
  "A.J. Soprano",
  "Uncle Junior",
  "Bobby Baccalieri",
  "Adriana La Cerva",
  "Vito Spatafore",

  // One Piece (additional)
  "Roronoa Zoro",
  "Nico Robin",
  "Franky",
  "Brook",
  "Usopp",
  "Shanks",
  "Portgas D. Ace",
  "Trafalgar Law",
  "Boa Hancock",

  // The Muppets
  "Kermit the Frog",
  "Miss Piggy",
  "Fozzie Bear",
  "Gonzo",
  "Animal",
  "Beaker",
  "Dr. Bunsen Honeydew",
  "Statler",
  "Waldorf",
  "Swedish Chef",
  "Rowlf",
  "Rizzo the Rat",
  "Pepe the Prawn",

  // Mission: Impossible
  "Ethan Hunt",
  "Benji Dunn",
  "Luther Stickell",
  "Ilsa Faust",
  "Julia Meade",

  // What We Do in the Shadows
  "Nandor the Relentless",
  "Laszlo Cravensworth",
  "Nadja",
  "Colin Robinson",
  "Guillermo",
  "The Guide",
  "Baron Afanas",

  // Outlander
  "Claire Fraser",
  "Jamie Fraser",
  "Brianna Fraser",
  "Roger MacKenzie",
  "Murtagh Fraser",
  "Dougal MacKenzie",
  "Geillis Duncan",

  // The Wire
  "Jimmy McNulty",
  "Stringer Bell",
  "Omar Little",
  "Avon Barksdale",
  "Bunk Moreland",
  "Lester Freamon",
  "Bubbles",
  "Marlo Stanfield",
  "Prop Joe",
  "Kima Greggs",
  "Bodie Broadus",
  "D'Angelo Barksdale",

  // Firefly
  "Malcolm Reynolds",
  "Zoe Washburne",
  "Hoban Washburne",
  "Inara Serra",
  "Jayne Cobb",
  "Kaylee Frye",
  "Simon Tam",
  "River Tam",
  "Shepherd Book",

  // Stand-up Comedians
  "George Carlin",
  "Richard Pryor",
  "Eddie Murphy",
  "Chris Rock",
  "Dave Chappelle",
  "Bill Burr",
  "Louis C.K.",
  "Jim Gaffigan",
  "John Mulaney",
  "Ali Wong",
  "Hasan Minhaj",
  "Trevor Noah",
  "Kevin Hart",
  "Amy Schumer",
  "Ellen DeGeneres",
  "Ricky Gervais",
  "Jimmy Carr",
  "Nate Bargatze",
  "Bert Kreischer",
  "Tom Segura",

  // It's Always Sunny in Philadelphia
  "Dennis Reynolds",
  "Mac McDonald",
  "Charlie Kelly",
  "Dee Reynolds",
  "Frank Reynolds",
  "The Waitress",
  "Cricket",
  "Artemis",

  // The West Wing
  "Jed Bartlet",
  "Leo McGarry",
  "Josh Lyman",
  "C.J. Cregg",
  "Toby Ziegler",
  "Sam Seaborn",
  "Donna Moss",
  "Charlie Young",
  "Abbey Bartlet",
  "Will Bailey",

  // Monty Python
  "John Cleese",
  "Eric Idle",
  "Terry Gilliam",
  "Terry Jones",
  "Michael Palin",
  "Graham Chapman",

  // Arrested Development
  "Michael Bluth",
  "George Bluth Sr.",
  "Lucille Bluth",
  "Gob Bluth",
  "Lindsay Bluth",
  "Tobias Fünke",
  "Maeby Fünke",
  "Buster Bluth",
  "George Michael Bluth",
  "Kitty Sanchez",
  "Lupe",

  // The X-Files
  "Fox Mulder",
  "Dana Scully",
  "Walter Skinner",
  "The Smoking Man",
  "Alex Krycek",
  "John Doggett",
  "Monica Reyes",

  // The IT Crowd
  "Roy Trenneman",
  "Maurice Moss",
  "Jen Barber",
  "Douglas Reynholm",
  "Richmond Avenal",

  // Schitt's Creek
  "Johnny Rose",
  "Moira Rose",
  "David Rose",
  "Alexis Rose",
  "Stevie Budd",
  "Patrick Brewer",
  "Roland Schitt",
  "Jocelyn Schitt",
  "Twyla Sands",

  // Silicon Valley
  "Richard Hendricks",
  "Erlich Bachman",
  "Dinesh Chugtai",
  "Bertram Gilfoyle",
  "Jared Dunn",
  "Monica Hall",
  "Gavin Belson",
  "Jian-Yang",
  "Big Head",

  // Foundation
  "Hari Seldon",
  "Gaal Dornick",
  "Salvor Hardin",
  "Hober Mallow",
  "Brother Day",
  "Brother Dusk",
  "Brother Dawn",
  "Demerzel",

  // Battlestar Galactica
  "William Adama",
  "Laura Roslin",
  "Kara Thrace",
  "Lee Adama",
  "Gaius Baltar",
  "Number Six",
  "Saul Tigh",
  "Sharon Valerii",
  "Karl Agathon",
  "Galen Tyrol",
  "Felix Gaeta",

  // Popular Scientists
  "Albert Einstein",
  "Isaac Newton",
  "Marie Curie",
  "Charles Darwin",
  "Nikola Tesla",
  "Galileo Galilei",
  "Stephen Hawking",
  "Carl Sagan",
  "Neil deGrasse Tyson",
  "Richard Feynman",
  "Jane Goodall",
  "Alan Turing",
  "Ada Lovelace",
  "Rosalind Franklin",
  "Edwin Hubble",
  "Niels Bohr",
  "Mae Jemison",
  "Katherine Johnson",
  "Rachel Carson",
  "Michio Kaku",

  // Futurama
  "Philip J. Fry",
  "Turanga Leela",
  "Bender",
  "Professor Farnsworth",
  "Amy Wong",
  "Hermes Conrad",
  "Dr. Zoidberg",
  "Zapp Brannigan",
  "Kif Kroker",
  "Nibbler",
  "Mom",
  "Calculon",

  // Full House
  "Danny Tanner",
  "Jesse Katsopolis",
  "Joey Gladstone",
  "D.J. Tanner",
  "Stephanie Tanner",
  "Michelle Tanner",
  "Rebecca Donaldson",
  "Kimmy Gibbler",
  "Steve Hale",
  "Nicky Katsopolis",
  "Alex Katsopolis",

  // That 70's Show
  "Eric Forman",
  "Donna Pinciotti",
  "Michael Kelso",
  "Jackie Burkhart",
  "Steven Hyde",
  "Fez",
  "Red Forman",
  "Kitty Forman",
  "Bob Pinciotti",
  "Midge Pinciotti",
  "Leo Chingkwake",

  // Boy Meets World
  "Cory Matthews",
  "Topanga Lawrence",
  "Shawn Hunter",
  "Eric Matthews",
  "Mr. Feeny",
  "Alan Matthews",
  "Amy Matthews",
  "Morgan Matthews",
  "Angela Moore",
  "Jack Hunter",

  // Family Matters
  "Steve Urkel",
  "Carl Winslow",
  "Harriette Winslow",
  "Eddie Winslow",
  "Laura Winslow",
  "Judy Winslow",
  "Richie Crawford",
  "Waldo Faldo",
  "Myra Monkhouse",

  // King of the Hill
  "Hank Hill",
  "Peggy Hill",
  "Bobby Hill",
  "Dale Gribble",
  "Bill Dauterive",
  "Boomhauer",
  "Luanne Platter",
  "Cotton Hill",
  "Kahn Souphanousinphone",
  "Minh Souphanousinphone",
  "Connie Souphanousinphone",

  // Modern Family
  "Phil Dunphy",
  "Claire Dunphy",
  "Haley Dunphy",
  "Alex Dunphy",
  "Luke Dunphy",
  "Jay Pritchett",
  "Gloria Pritchett",
  "Manny Delgado",
  "Cameron Tucker",
  "Mitchell Pritchett",
  "Lily Tucker-Pritchett",

  // Scrubs
  "J.D. Dorian",
  "Turk",
  "Elliot Reid",
  "Carla Espinosa",
  "Dr. Cox",
  "Dr. Kelso",
  "The Janitor",
  "Todd Quinlan",
  "Ted Buckland",
  "Laverne Roberts",

  // Malcolm in the Middle
  "Malcolm",
  "Hal",
  "Lois",
  "Francis",
  "Reese",
  "Dewey",
  "Jamie",
  "Stevie Kenarban",
  "Craig Feldspar",

  // 30 Rock
  "Liz Lemon",
  "Jack Donaghy",
  "Tracy Jordan",
  "Jenna Maroney",
  "Kenneth Parcell",
  "Pete Hornberger",
  "Frank Rossitano",
  "Cerie Xerox",
  "Dr. Leo Spaceman",

  // The Big Bang Theory
  "Sheldon Cooper",
  "Leonard Hofstadter",
  "Penny",
  "Howard Wolowitz",
  "Raj Koothrappali",
  "Amy Farrah Fowler",
  "Bernadette Rostenkowski",
  "Stuart Bloom",
  "Barry Kripke",
  "Leslie Winkle",

  // Saved by the Bell
  "Zack Morris",
  "Kelly Kapowski",
  "A.C. Slater",
  "Jessie Spano",
  "Lisa Turtle",
  "Screech Powers",
  "Mr. Belding",
  "Tori Scott",
];

/**
 * Cache of userId -> character name mappings for the current session.
 * This gets regenerated on page reload/session reset.
 *
 * Purpose: Ensures consistent character name assignment for the same user
 * within a single session, while allowing re-randomization on refresh.
 *
 * @type {Map<string, string>}
 * @example
 * // User "abc-123" -> "Iron Man" (consistent within session)
 * sessionMappingCache.get("abc-123") // returns "Iron Man"
 */
const sessionMappingCache = new Map<string, string>();

/**
 * Track which character names have been used in this session to prevent duplicates.
 *
 * Purpose: Guarantees uniqueness of pseudonyms across all users in the current session.
 * No two users will have the same character name displayed simultaneously.
 *
 * @type {Set<string>}
 */
const usedCharacterNames = new Set<string>();

/**
 * Available character names pool (shuffled each session).
 * This is initialized once per session and used to assign unique names.
 *
 * Purpose: Randomizes the order of name assignment to ensure different
 * distributions across sessions. The shuffle happens once per page load.
 *
 * @type {string[]}
 */
let availableNamesPool: string[] = [];

/**
 * Counter for handling overflow when we have more users than character names.
 *
 * Purpose: Tracks how many times we've exceeded the base pool size,
 * used to generate suffixes like "II", "III", etc.
 *
 * @type {number}
 */
let overflowCounter = 0;

/**
 * Fisher-Yates shuffle algorithm to randomize the character names array.
 *
 * This ensures an unbiased random distribution of character names each session.
 * Time complexity: O(n), where n is the array length.
 *
 * @template T - The type of array elements
 * @param {T[]} array - The array to shuffle
 * @returns {T[]} A new shuffled copy of the array
 *
 * @example
 * shuffleArray(['A', 'B', 'C']) // might return ['C', 'A', 'B']
 */
function shuffleArray<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Initialize the available names pool with a shuffled version of character names.
 *
 * This function is called once per session (lazy initialization) to create
 * the randomized pool from which names are drawn sequentially.
 *
 * @returns {void}
 */
function initializeNamesPool(): void {
  if (availableNamesPool.length === 0) {
    availableNamesPool = shuffleArray(CHARACTER_NAMES);
  }
}

/**
 * Generate a unique character name for a user ID.
 * Each user gets a unique character name within the session.
 * If there are more users than character names, suffixes are added (e.g., "Iron Man II").
 *
 * @param userId The user identifier
 * @returns A unique random character name
 */
export function getRandomCharacterName(userId: string): string {
  // Check if we already have a mapping for this session
  if (sessionMappingCache.has(userId)) {
    return sessionMappingCache.get(userId)!;
  }

  // Initialize the names pool if not already done
  initializeNamesPool();

  let characterName = "";

  // Find an unused name from the pool
  if (usedCharacterNames.size < CHARACTER_NAMES.length) {
    // We still have unused names available
    for (const name of availableNamesPool) {
      if (!usedCharacterNames.has(name)) {
        characterName = name;
        break;
      }
    }

    // This should never happen, but just in case
    if (!characterName) {
      characterName = availableNamesPool[usedCharacterNames.size % availableNamesPool.length];
    }
  } else {
    // We've run out of unique names, start adding suffixes
    overflowCounter++;
    const baseNameIndex = (overflowCounter - 1) % CHARACTER_NAMES.length;
    const suffix = Math.floor((overflowCounter - 1) / CHARACTER_NAMES.length) + 2;

    // Convert number to Roman numerals for a more stylish look
    const romanNumerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X'];
    const romanSuffix = suffix <= 10 ? romanNumerals[suffix - 1] : suffix.toString();

    characterName = `${availableNamesPool[baseNameIndex]} ${romanSuffix}`;
  }

  // Mark this name as used and cache the mapping
  usedCharacterNames.add(characterName);
  sessionMappingCache.set(userId, characterName);

  return characterName;
}

/**
 * Clear the session mapping cache. This will cause new character names
 * to be generated for all users on the next call to getRandomCharacterName.
 */
export function clearCharacterNameCache(): void {
  sessionMappingCache.clear();
  usedCharacterNames.clear();
  availableNamesPool = [];
  overflowCounter = 0;
}

/**
 * Get the number of cached character name mappings.
 */
export function getCharacterNameCacheSize(): number {
  return sessionMappingCache.size;
}
