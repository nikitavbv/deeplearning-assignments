use {
    std::{fs::{self, read_dir, read_to_string}, collections::HashMap},
    rand::prelude::*,
    tch::{Tensor, Kind, data::Iter2, vision::imagenet::load_image_and_resize224},
};

// Dataset downloaded from https://huggingface.co/datasets/imagenet-1k/tree/main/data

pub fn prepare_datasets() {
    println!("preparing datasets");

    let classes_to_use: HashMap<String, usize> = vec![
        "n01440764", //: "tench, Tinca tinca"
        "n01443537", //: "goldfish, Carassius auratus"
        "n01484850", //: "great white shark, white shark, man-eater, man-eating shark, Carcharodon carcharias"
        "n01491361", //: "tiger shark, Galeocerdo cuvieri"
        "n01494475", //: "hammerhead, hammerhead shark"
        "n01496331", //: "electric ray, crampfish, numbfish, torpedo"
        "n01498041", //: "stingray"
        "n01514668", //: "cock"
        "n01514859", //: "hen"
        "n01518878", //: "ostrich, Struthio camelus"
        "n01530575", //: "brambling, Fringilla montifringilla"
        "n01531178", //: "goldfinch, Carduelis carduelis"
        "n01532829", //: "house finch, linnet, Carpodacus mexicanus"
        "n01534433", //: "junco, snowbird"
        "n01537544", //: "indigo bunting, indigo finch, indigo bird, Passerina cyanea"
        "n01558993", //: "robin, American robin, Turdus migratorius"
        "n01560419", //: "bulbul"
        "n01580077", //: "jay"
        "n01582220", //: "magpie"
        "n01592084", //: "chickadee"
        "n01601694", //: "water ouzel, dipper"
        "n01608432", //: "kite"
        "n01614925", //: "bald eagle, American eagle, Haliaeetus leucocephalus"
        "n01616318", //: "vulture"
        "n01622779", //: "great grey owl, great gray owl, Strix nebulosa"
        "n01629819", //: "European fire salamander, Salamandra salamandra"
        "n01630670", //: "common newt, Triturus vulgaris"
        "n01631663", //: "eft"
        "n01632458", //: "spotted salamander, Ambystoma maculatum"
        "n01632777", //: "axolotl, mud puppy, Ambystoma mexicanum"
        "n01641577", //: "bullfrog, Rana catesbeiana"
        "n01644373", //: "tree frog, tree-frog"
        "n01644900", //: "tailed frog, bell toad, ribbed toad, tailed toad, Ascaphus trui"
        "n01664065", //: "loggerhead, loggerhead turtle, Caretta caretta"
        "n01665541", //: "leatherback turtle, leatherback, leathery turtle, Dermochelys coriacea"
        "n01667114", //: "mud turtle"
        "n01667778", //: "terrapin"
        "n01669191", //: "box turtle, box tortoise"
        "n01675722", //: "banded gecko"
        "n01677366", //: "common iguana, iguana, Iguana iguana"
        "n01682714", //: "American chameleon, anole, Anolis carolinensis"
        "n01685808", //: "whiptail, whiptail lizard"
        "n01687978", //: "agama"
        "n01688243", //: "frilled lizard, Chlamydosaurus kingi"
        "n01689811", //: "alligator lizard"
        "n01692333", //: "Gila monster, Heloderma suspectum"
        "n01693334", //: "green lizard, Lacerta viridis"
        "n01694178", //: "African chameleon, Chamaeleo chamaeleon"
        "n01695060", //: "Komodo dragon, Komodo lizard, dragon lizard, giant lizard, Varanus komodoensis"
        "n01697457", //: "African crocodile, Nile crocodile, Crocodylus niloticus"
        "n01698640", //: "American alligator, Alligator mississipiensis"
        "n01704323", //: "triceratops"
        "n01728572", //: "thunder snake, worm snake, Carphophis amoenus"
        "n01728920", //: "ringneck snake, ring-necked snake, ring snake"
        "n01729322", //: "hognose snake, puff adder, sand viper"
        "n01729977", //: "green snake, grass snake"
        "n01734418", //: "king snake, kingsnake"
        "n01735189", //: "garter snake, grass snake"
        "n01737021", //: "water snake"
        "n01739381", //: "vine snake"
        "n01740131", //: "night snake, Hypsiglena torquata"
        "n01742172", //: "boa constrictor, Constrictor constrictor"
        "n01744401", //: "rock python, rock snake, Python sebae"
        "n01748264", //: "Indian cobra, Naja naja"
        "n01749939", //: "green mamba"
        "n01751748", //: "sea snake"
        "n01753488", //: "horned viper, cerastes, sand viper, horned asp, Cerastes cornutus"
        "n01755581", //: "diamondback, diamondback rattlesnake, Crotalus adamanteus"
        "n01756291", //: "sidewinder, horned rattlesnake, Crotalus cerastes"
        "n01768244", //: "trilobite"
        "n01770081", //: "harvestman, daddy longlegs, Phalangium opilio"
        "n01770393", //: "scorpion"
        "n01773157", //: "black and gold garden spider, Argiope aurantia"
        "n01773549", //: "barn spider, Araneus cavaticus"
        "n01773797", //: "garden spider, Aranea diademata"
        "n01774384", //: "black widow, Latrodectus mactans"
        "n01774750", //: "tarantula"
        "n01775062", //: "wolf spider, hunting spider"
        "n01776313", //: "tick"
        "n01784675", //: "centipede"
        "n01795545", //: "black grouse"
        "n01796340", //: "ptarmigan"
        "n01797886", //: "ruffed grouse, partridge, Bonasa umbellus"
        "n01798484", //: "prairie chicken, prairie grouse, prairie fowl"
        "n01806143", //: "peacock"
        "n01806567", //: "quail"
        "n01807496", //: "partridge"
        "n01817953", //: "African grey, African gray, Psittacus erithacus"
        "n01818515", //: "macaw"
        "n01819313", //: "sulphur-crested cockatoo, Kakatoe galerita, Cacatua galerita"
        "n01820546", //: "lorikeet"
        "n01824575", //: "coucal"
        "n01828970", //: "bee eater"
        "n01829413", //: "hornbill"
        "n01833805", //: "hummingbird"
        "n01843065", //: "jacamar"
        "n01843383", //: "toucan"
        "n01847000", //: "drake"
        "n01855032", //: "red-breasted merganser, Mergus serrator"
        "n01855672", //: "goose"
        "n01860187", //: "black swan, Cygnus atratus"
        "n01871265", //: "tusker"
    ].iter().enumerate().map(|v| (v.1.to_string(), v.0)).collect();
    
    prepare_dataset(&classes_to_use, "train");
    prepare_dataset(&classes_to_use, "val");
}

fn prepare_dataset(classes_to_use: &HashMap<String, usize>, dataset: &str) {
    println!("preparing dataset: {}", dataset);

    let total_classes = classes_to_use.len();

    let mut images = Vec::new();
    let mut labels = Vec::new();
    let mut chunk_index = 0;

    let max_chunk_size = 5000;

    let mut total = 0;

    for file in read_dir(format!("imagenet/raw/{}", dataset)).unwrap() {
        let file = file.unwrap();
        let name = file.file_name().to_string_lossy().to_string().replace(".JPEG", "");

        let class = name.split("_").last().unwrap();
        let class_index = classes_to_use.get(class);

        if let Some(class_index) = class_index {
            let image = load_image_and_resize224(file.path()).unwrap().unsqueeze(0);
            images.push(image);

            let mut label = vec![0.0; total_classes];
            label[*class_index] =  1.0;
            let label = Tensor::from_slice(&label).to_kind(Kind::Float).unsqueeze(0);
            labels.push(label);
            
            total += 1;
        
            if images.len() > max_chunk_size {
                save_chunk(dataset, chunk_index, &images, &labels);
                chunk_index += 1;

                images.clear();
                labels.clear();
            }
        }
    }

    if images.len() > 0 {
        save_chunk(dataset, chunk_index, &images, &labels);
        fs::write(format!("./imagenet/dataset/{}/total_chunks", dataset), (chunk_index + 1).to_string()).unwrap();
    }

    println!("done, total entries: {}", total);
}

fn save_chunk(dataset: &str, chunk_index: i64, images: &[Tensor], labels: &[Tensor]) {
    let images = Tensor::cat(&images, 0);
    let labels = Tensor::cat(&labels, 0);

    images.save(format!("./imagenet/dataset/{}/images_{}.pt", dataset, chunk_index)).unwrap();
    labels.save(format!("./imagenet/dataset/{}/labels_{}.pt", dataset, chunk_index)).unwrap();
}

pub struct Dataset {
    name: String,
    chunks: Vec<u64>,
    batch_size: i64,
}

impl Dataset {
    pub fn new(name: &str, batch_size: i64) -> Self {
        let total_chunks: u64 = read_to_string(format!("./imagenet/dataset/{}/total_chunks", name)).unwrap().parse().unwrap();
        let mut chunks: Vec<u64> = (0..total_chunks).collect();
        chunks.shuffle(&mut rand::thread_rng());

        Self {
            name: name.to_owned(),
            chunks,
            batch_size,
        }
    }

    pub fn has_more_chunks(&self) -> bool {
        !self.chunks.is_empty()
    }

    pub fn next_chunk(&mut self) -> Iter2 {
        let chunk_index = self.chunks.pop().unwrap();
        let xs = Tensor::load(format!("./imagenet/dataset/{}/images_{}.pt", &self.name, chunk_index)).unwrap();
        let ys = Tensor::load(format!("./imagenet/dataset/{}/labels_{}.pt", &self.name, chunk_index)).unwrap();
        let mut iter = Iter2::new(&xs, &ys, self.batch_size);
        iter.return_smaller_last_batch();
        iter.shuffle();
        iter
    }
}
