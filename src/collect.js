const fs = require("fs")
const request = require("request")
const puppeteer = require('puppeteer')
const uuidv4 = require('uuid/v4')
const { execSync} = require('child_process')


const TWTICH_DIRECTORY_URL = 'https://www.twitch.tv/directory/'
const TWTICH_BASE_URL = 'https://www.twitch.tv/directory/game/'
const LOL_TWITCH = 'https://www.twitch.tv/directory/game/League%20of%20Legends'
const MTGA_TWITCH = 'https://www.twitch.tv/directory/game/Magic%3A%20The%20Gathering'
const SMASH_TWITCH = 'https://www.twitch.tv/directory/game/Super%20Smash%20Bros.%20Ultimate'

const MTGA_YOUTUBE = 'UCE04gbPEl9kD5IHTBMmK0yw'
const LOL_YOUTUBE =  'UCZtmNrG53nmbq-Ww2VJrxEQ'

const puppeteer_config = {
  headless: true,
  defaultViewport: {
    width: 4800,
    height: 3000
  },
  // args: ['--start-fullscreen']
}


const getGameList = async(total_games = 10) => {
  const browser = await puppeteer.launch(puppeteer_config)
  const page = await browser.newPage()
  await page.goto("https://www.twitch.tv/directory")
  await page.waitForSelector('.tw-tower')

  let game_urls = await page.evaluate(()=>{
    let game_urls = []
    let children = document.querySelector('.tw-tower').children
    for (var i = 0; i < children.length; i++) {
      let child = children[i]
      if(child.querySelector('.tw-box-art-card__link')){
        game_urls.push(child.querySelector('.tw-box-art-card__link').href)
      }
    }
    return game_urls
  })

  return game_urls.map(url => {
    let decoded = decodeURI(url).replaceAll("%3A", ":").replaceAll("%2F", "/").replaceAll("%26", "and")
    let name = decoded.split("https://www.twitch.tv/directory/game/")[1]
    return {
      url,
      name,
      dir:  "./images/" + name.replaceAll(" ", "_").replaceAll("/", "+"),
    }
  }).filter(game =>{
    return game.name != "Just Chatting"
  }).slice(0, total_games)
}


const collectGame = async(game, page)=>{
  await page.goto(game.url)
  await page.waitForSelector('.tw-tower')

  let image_urls = await page.evaluate(()=>{
    let image_urls = []
    let children = document.querySelector('.tw-tower').children
    for (var i = 0; i < children.length; i++) {
      let child = children[i]
      if(child.querySelector('img')){
        image_urls.push(child.querySelector('img').src)
      }
    }
    return image_urls
  })

  if (!fs.existsSync(game.dir)){
    fs.mkdirSync(game.dir)
  }

  console.log("total image_urls: ", image_urls.length)
  for(let i=0; i<image_urls.length; i++){
    let url = image_urls[i]
    let file_name = game.dir + "/" + uuidv4() + url.replaceAll("/", '+')
    const child = execSync(`wget ${url} -O ${file_name}`)
  }

  // image_urls.forEach(url => {
  //   download(url, game.dir + "/" + uuidv4() + url.replace(re, '+'), function() {
  //   })
  // })

  return
}


function download(uri, filename, callback) {
  request.head(uri, function(err, res, body) {
    request(uri)
    .pipe(fs.createWriteStream(filename))
    .on("close", callback)
 });
}


String.prototype.replaceAll = function(search, replacement) {
    var target = this
    return target.split(search).join(replacement)
};

const main = async()=>{
  try {
    const browser = await puppeteer.launch(puppeteer_config)
    const games = await getGameList(10)

    let promises = []
    for(let i=0; i<games.length; i++){
      let page = await browser.newPage()
      promises.push(collectGame(games[i], page))
    }

    await Promise.all(promises)
    await browser.close()
    console.log("collection complete")
  } catch(e){
    console.log("error: ", e)
  }
}


main()
setInterval(main, 1000 * 60 * 10)
