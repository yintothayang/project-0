const fs = require("fs")
const request = require("request")
const puppeteer = require('puppeteer')
const uuidv4 = require('uuid/v4')

const LOL_TWITCH = 'https://www.twitch.tv/directory/game/League%20of%20Legends'
const MTGA_TWITCH = 'https://www.twitch.tv/directory/game/Magic%3A%20The%20Gathering'
const SMASH_TWITCH = 'https://www.twitch.tv/directory/game/Super%20Smash%20Bros.%20Ultimate'

const MTGA_YOUTUBE = 'UCE04gbPEl9kD5IHTBMmK0yw'
const LOL_YOUTUBE =  'UCZtmNrG53nmbq-Ww2VJrxEQ'

const puppeteer_config = {
  headless: false,
  defaultViewport: {
    width: 1800,
    height: 1000
  },
  args: ['--start-fullscreen']
}



const collectGame = async(url, dir)=>{
  const browser = await puppeteer.launch(puppeteer_config)
  const page = await browser.newPage()
  // await page.setViewport({ width: 1900, height: 800 })

  await page.goto(url)
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

  await browser.close()
  console.log(image_urls)

  var find = '/'
  var re = new RegExp(find, 'g')
  image_urls.forEach(url => {
    download(url, "./images/twitch/" + dir + "/" + uuidv4() + url.replace(re, '+'), function() {
      console.log("Image downloaded");
    });
  })
}


const main = async()=>{
  console.log("start collect")
  let promises = [
    collectGame(LOL_TWITCH, 'lol'),
    collectGame(MTGA_TWITCH, 'mtga'),
    collectGame(SMASH_TWITCH, 'smash')
  ]
  await Promise.all(promises)
  console.log("end collect")
}


function download(uri, filename, callback) {
  request.head(uri, function(err, res, body) {
    request(uri)
    .pipe(fs.createWriteStream(filename))
    .on("close", callback);
 });
}


main()
setInterval(main, 1000 * 60 * 10)
