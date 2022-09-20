import { yamlUpdateSocket } from './utils.js';

const yamlLoader = document.querySelector('#yaml-loader');

yamlLoader.onclick = function(e) {
  const yamlFileLocator = document.querySelector('#yaml-file-locator');
  yamlUpdateSocket.send(JSON.stringify({
      'yaml_file': yamlFileLocator.value,
  }));
};
