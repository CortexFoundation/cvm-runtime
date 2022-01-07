import { yamlUpdateSocket } from './utils.js';

document.querySelector('#yaml-loader').onclick = function(e) {
  const yamlFileLocator = document.querySelector('#yaml-file-locator');
  yamlUpdateSocket.send(JSON.stringify({
      'yaml_file': yamlFileLocator.value,
  }));
};
