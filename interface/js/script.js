var start = false;
var startExploration;

var activeIndex = -1;
var selectedRegionIndex = -1
var currentPipeline = [];
var currentOperatorResults = [];
var currentDataset = document.getElementById("dataset").value;

class TreeNode {
  constructor(value, userCount, movieCount, fdr, power, coverage, aggregation, currentNumber) {
    this.value = value;
    this.userCount = userCount;
    this.movieCount = movieCount;
    this.fdr = fdr;
    this.power = power;
    this.coverage = coverage;
    this.aggregation = aggregation;
    this.currentNumber = currentNumber;
    this.children = [];
  }
}

function nodeToJSON (node, parent, n) {
  let inputDataRegion = [];
  let outputDataRegion = node.value.split('_');
  let name;
  let JSONTree;

  if (parent != null) {
    inputDataRegion = parent.value.split('_');
  }

  // Difference between two arrays
  name = outputDataRegion.filter(x => !inputDataRegion.includes(x))[0];
  name = name.charAt(0).toUpperCase() + name.slice(1);
  JSONTree = {
    "name": name,
    "children": []
  };

  //console.log(node);

  if (node.currentNumber != null) {
    JSONTree["hypothesis"] = "H" + String(node.currentNumber + 1);
  } else {
    JSONTree["hypothesis"] = null;
  }
  

  if ((node.userCount != null) && (node.movieCount != null)) {
    JSONTree["count"] = `${node.userCount} ${(node.userCount == 1 ? 'user' : 'users')}, ${node.movieCount} ${(node.movieCount == 1 ? 'movie' : 'movies')}`;
  } else {
    JSONTree["count"] = null;
  }
  
  if ((node.fdr != null) && (node.power != null) && (node.coverage != null)) {
    JSONTree["stats"] = `FDR: ${node.fdr} | Power: ${node.power} | Coverage: ${node.coverage}`;
    JSONTree["fdr"] = node.fdr;
    JSONTree["power"] = node.power;
    JSONTree["coverage"] = node.coverage;
    n++;
  } else {
    JSONTree["stats"] = null;
    JSONTree["fdr"] = null;
    JSONTree["power"] = null;
    JSONTree["coverage"] = null;
  }

  for (let i = 0; i < node.children.length; i++) {
    JSONTree["children"].push(nodeToJSON(node.children[i], node, n));
  }

  return JSONTree;
}

function formatDataRegion(dataRegion) {
  let genre_list =  ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'];
  let formattedDataRegion;

  if (dataRegion == 'MovieLens') {
    formattedDataRegion = dataRegion;
  }
  else if (dataRegion == 'M') {
    formattedDataRegion = 'Male users';
  } else if (dataRegion == 'F') {
    formattedDataRegion = 'Female users';
  } else if ((dataRegion == "Long") || (dataRegion == "Short") || (dataRegion.indexOf("Very") != -1)) {
    formattedDataRegion = 'Movies of ' + dataRegion + ' duration';
  } else if (genre_list.includes(dataRegion)) {
    formattedDataRegion = dataRegion + ' movies';
  } else if (dataRegion.indexOf("'") != -1) {
    formattedDataRegion = 'Movies released in the ' + dataRegion;
  } else if ((dataRegion == "Under 18") || (dataRegion == "<18")) {
    formattedDataRegion = 'Users under the age of 18';
  } else if (dataRegion == ">56") {
    formattedDataRegion = 'Users above the age of 56';
  } else if ((dataRegion == "18-24") || (dataRegion.indexOf("5-") != -1) || (dataRegion[0] == "5")) {
    formattedDataRegion = 'Users of the age ' + dataRegion;
  } else {
    formattedDataRegion = 'Users that work as ' + dataRegion;
  }

  return formattedDataRegion;
}

function nodeToJSON2 (node, parent, n, childAggregation) {
  let inputDataRegion = [];
  let outputDataRegion = node.value.split('_');
  let name;
  let JSONTree;

  if (parent != null) {
    inputDataRegion = parent.value.split('_');
  }

  // Difference between two arrays
  name = outputDataRegion.filter(x => !inputDataRegion.includes(x))[0];
  name = name.charAt(0).toUpperCase() + name.slice(1);
  JSONTree = {
    "key": formatDataRegion(name),
    "children": []
  };

  //console.log(node);

  /*
  if (node.aggregation != null) {
    if (node.aggregation == 'mean') {
      JSONTree["hypothesis"] = "μ > 2.5";
    } else if (node.aggregation == 'variance') {
      JSONTree["hypothesis"] = "σ² > 2.5";
    } else {
      JSONTree["hypothesis"] = "R ≁ U";
    }
  } else {
    JSONTree["hypothesis"] = null;
  }
  */
  if (childAggregation != undefined) {
    if (childAggregation == 'mean') {
      JSONTree["hypothesis"] = "μ > 2.5";
    } else if (childAggregation == 'variance') {
      JSONTree["hypothesis"] = "σ² > 2.5";
    } else {
      JSONTree["hypothesis"] = "R ≁ U";
    }
  } else {
    JSONTree["hypothesis"] = null;
  }

  if ((node.userCount != null) && (node.movieCount != null)) {
    JSONTree["key"] += ` (${node.userCount} ${(node.userCount == 1 ? 'user' : 'users')}, ${node.movieCount} ${(node.movieCount == 1 ? 'movie' : 'movies')})`;
    JSONTree["users"] = node.userCount;
    JSONTree["movies"] = node.movieCount;
  }
  
  if ((node.fdr != null) && (node.power != null) && (node.coverage != null)) {
    JSONTree["stats"] = `FDR: ${node.fdr} | Power: ${node.power} | Coverage: ${node.coverage}`;
    JSONTree["fdr"] = node.fdr;
    JSONTree["power"] = node.power;
    JSONTree["coverage"] = node.coverage;
    n++;
  } else {
    JSONTree["stats"] = null;
    JSONTree["fdr"] = null;
    JSONTree["power"] = null;
    JSONTree["coverage"] = null;
  }

  for (let i = 0; i < node.children.length; i++) {
    let userCount, movieCount;

    if (node.children[i].userCount == null) {
      console.log('Children count', node.children.length);
      console.log(currentPipeline[node.currentNumber]);
      console.log(currentOperatorResults[node.currentNumber]);
      //console.log(node.value, i, currentOperatorResults[node.currentNumber].length);
      //console.log(currentOperatorResults[node.currentNumber][i+1]);
      if (currentOperatorResults[node.currentNumber].length > i+1) {
        userCount = String(currentOperatorResults[node.currentNumber][i+1]['genders'].reduce(add, 0));
        movieCount = String(currentOperatorResults[node.currentNumber][i+1]['genres'].reduce(add, 0));
      } else {
        userCount = String(currentOperatorResults[node.currentNumber][i]['genders'].reduce(add, 0));
        movieCount = String(currentOperatorResults[node.currentNumber][i]['genres'].reduce(add, 0));
      }
      
      node.children[i].userCount = userCount;
      node.children[i].movieCount = movieCount;
    }

    JSONTree["children"].push(nodeToJSON2(node.children[i], node, n, node.aggregation));
  }

  return JSONTree;
}

function currentPipelineToTree() {
  var tree = new TreeNode('MovieLens', 6040, 3952, null, null, null, null, null);

  function explore(value, childrenValues, userCount, movieCount, fdr, power, coverage, aggregation, currentNumber) {
    let newNode = new TreeNode(value, userCount, movieCount, fdr, power, coverage, aggregation, currentNumber);

    for (i = 0; i < childrenValues.length; i++) {
      newNode.children.push(new TreeNode(childrenValues[i], null, null, null, null, null, null, null))
    }

    tree.children.push(newNode);

    return newNode;
  }

  function exploit(possibleCurrentNodes, value, childrenValues, userCount, movieCount, fdr, power, coverage, aggregation, currentNumber) {
    let currentNode;
    let i;

    for (i = 0; i < possibleCurrentNodes.length; i++) {
      if (possibleCurrentNodes[i].value == value) {
        currentNode = possibleCurrentNodes[i];
        break;
      }
    }

    currentNode.userCount = userCount;
    currentNode.movieCount = movieCount;
    currentNode.fdr = fdr;
    currentNode.power = power;
    currentNode.coverage = coverage;
    currentNode.aggregation = aggregation;
    currentNode.currentNumber = currentNumber;

    let fromIndex = i; // 👉️ 0
    let toIndex = 0;

    let element = possibleCurrentNodes.splice(fromIndex, 1)[0];
    possibleCurrentNodes.splice(toIndex, 0, element);

    for (i = 0; i < childrenValues.length; i++) {
      currentNode.children.push(new TreeNode(childrenValues[i], null, null, null, null, null, null, null))
    }

    return currentNode;
  }

  let explorationTree;

  for (let i = 0; i < currentPipeline.length; i++) {
    if ((i == 0) || (currentPipeline[i]['action'] == 'Explore')) {
      explorationTree = explore(
        currentPipeline[i]['input_data_region'],
        currentPipeline[i]['output_data_regions'],
        currentPipeline[i]['user_count'],
        currentPipeline[i]['movie_count'],
        currentPipeline[i]['fdr'],
        currentPipeline[i]['power'],
        currentPipeline[i]['coverage'],
        currentPipeline[i]['aggregation'],
        i
      );
    } else {
      explorationTree = exploit(
        explorationTree.children,
        currentPipeline[i]['input_data_region'],
        currentPipeline[i]['output_data_regions'],
        currentPipeline[i]['user_count'],
        currentPipeline[i]['movie_count'],
        currentPipeline[i]['fdr'],
        currentPipeline[i]['power'],
        currentPipeline[i]['coverage'],
        currentPipeline[i]['aggregation'],
        i
      );
    }

  }

  // console.log(JSON.stringify(nodeToJSON2(tree, null, 1)));

  const dataSpec = {
    source: nodeToJSON2(tree, null, 1)
  };

  const myChart = d3.indentedTree(dataSpec)
    //.propagateValue("users")
    .linkLabel("hypothesis", {onTop: false, align: "middle"})
    .nodeLabelLength(100)
  ;

  myChart.linkWidth(70);
  myChart.linkHeight(30);
  myChart.nodeTitle("stats");

  d3.select("#pills-profile").selectAll("div").remove();

  showChart(myChart);
  
  function showChart(_chart) {
    d3.select("#pills-profile")
      .append("div")
      .attr("class", "chart overflow-auto")
      .call(_chart);
  }
}

function add(accumulator, a) {
  return accumulator + a;
}

function currentPipelineToCSV() {
  //Test, Users, Movies, Aggregation, Null Value, FDR, Power, Coverage, Sex, Age, Occupation, Movie Genre, Movie Length, Decades
  csvContent = 'Test, Users, Movies, Aggregation, Null Value, FDR, Power, Coverage, Sex, Age, Occupation, Movie Genre, Movie Length, Decades\n';
  for (let i = 0; i < currentPipeline.length; i++) {
    if (i == 49) {
      console.log(currentPipeline[i]['coverage']);
      console.log(String(currentPipeline[i]['coverage']));
    }
    //console.log(currentPipeline[i]);
    //console.log(currentOperatorResults[i]);
    let test = 'H' + String(i);
    let users = String(currentPipeline[i]['user_count']);
    let movies = String(currentPipeline[i]['movie_count']);
    let aggregation = currentPipeline[i]['aggregation'];
    
    let null_value;
    if (currentPipeline[i]['aggregation'] == 'distribution') {
      null_value = 'uniform';
    } else {
      null_value = '2.5';
    }

    let fdr = String(currentPipeline[i]['fdr']);
    let power = String(currentPipeline[i]['power']);
    let coverage = String(currentPipeline[i]['coverage']);

    let sex = '', age = '', occupation = '', movie_genre = '', movie_length = '', decades = '';
    let genre_list =  ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'];
    
    let input_region_filters = currentPipeline[i]['input_data_region'].split('_');
    for (let j = 0; j < input_region_filters.length; j++) {
      filter = input_region_filters[j];

      
      
      if ((filter == 'M') || (filter == 'F')) {
        sex = filter;
      } else if ((filter == "Long") || (filter == "Short") || (filter.indexOf("Very") != -1)) {
        movie_length = filter;
      } else if (genre_list.includes(filter)) {
        movie_genre = filter;
      } else if (filter.indexOf("'") != -1) {
        decades = filter;
      } else if ((filter == "Under 18") || (filter == "18-24") || (filter.indexOf("5-") != -1) || (filter[0] == "5")) {
        age = filter;
      } else {
        occupation = filter;
      }
    }

    if (i == 49) {
      console.log(currentPipeline[i]['coverage']);
      console.log(String(currentPipeline[i]['coverage']));
    }
    
    csvContent += test + ',' + users + ',' + movies + ',' + aggregation + ',' + null_value + ',' + fdr + ',' + power + ',' + coverage + ',' + sex + ',' + age + ',' + occupation + ',' + movie_genre + ',' + movie_length + ',' + decades + '\n';

    let outputDataRegions = currentPipeline[i]['output_data_regions'];
    for (let dataRegionIndex = 0; dataRegionIndex < currentOperatorResults[i].length - 1; dataRegionIndex++) {
      test = 'DR';
      //console.log('Here');
      //console.log(outputDataRegions.length - 1);
      //console.log(currentOperatorResults[i]);
      users = String(currentOperatorResults[i][dataRegionIndex+1]['genders'].reduce(add, 0));
      movies = String(currentOperatorResults[i][dataRegionIndex+1]['genres'].reduce(add, 0));

      aggregation = '';
      null_value = '';
      fdr = '';
      power = '';
      coverage = '';

      sex = '';
      age = '';
      occupation = '';
      movie_genre = '';
      movie_length = '';
      decades = '';
      let region = currentPipeline[i]['output_data_regions'][dataRegionIndex].split('_');
      for (let j = 0; j < region.length; j++) {
        filter = region[j];
        
        if ((filter == 'M') || (filter == 'F')) {
          sex = filter;
        } else if ((filter == "Long") || (filter == "Short") || (filter.indexOf("Very") != -1)) {
          movie_length = filter;
        } else if (genre_list.includes(filter)) {
          movie_genre = filter;
        } else if (filter.indexOf("'") != -1) {
          decades = filter;
        } else if ((filter == "Under 18") || (filter == "18-24") || (filter.indexOf("5-") != -1) || (filter[0] == "5")) {
          age = filter;
        } else {
          occupation = filter;
        }
      }
      csvContent += test + ',' + users + ',' + movies + ',' + aggregation + ',' + null_value + ',' + fdr + ',' + power + ',' + coverage + ',' + sex + ',' + age + ',' + occupation + ',' + movie_genre + ',' + movie_length + ',' + decades + '\n';
    }
    // console.log(currentPipeline[i]);
  }
  console.log(csvContent);
}

function selectHypothesis(hypothesisIndex) {
  activeIndex = hypothesisIndex;
  selectRegion(0);
}

function selectRegion(regionIndex) {
  selectedRegionIndex = regionIndex;
}

async function callAgent(dataset, policy, selectedGroup, listRegions, aggregation) {
    var requestBody = {};

    if (dataset !== undefined) {
      requestBody["dataset"] = dataset;
    }

    if (policy !== undefined) {
        requestBody["policy"] = policy;
    }

    if (selectedGroup !== undefined) {
        requestBody["selected_group"] = selectedGroup;
    }

    if (listRegions !== undefined) {
        requestBody["list_regions"] = listRegions;
    }

    if (aggregation !== undefined) {
        requestBody["aggregation"] = aggregation;
    }

    const response = await fetch('http://127.0.0.1:5000/', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
    });

    // waits until the request completes...
    const data = await response.json();

    return data;
}

function updateBreadcrumbs(){
  let breadcrumbsElement = document.querySelector('#breadcrumbElements');
  let html = '';
  if (currentOperatorResults.length > 0) {
    for (let i = 0; i < currentOperatorResults[activeIndex][selectedRegionIndex]['breadcrumbs'].length; i++) {
      html += `<li class="breadcrumb-item active">${currentOperatorResults[activeIndex][selectedRegionIndex]['breadcrumbs'][i]}</li>`;
    }
  } else {
    html += `<li class="breadcrumb-item active">Empty pipeline</li>`;
  }
  breadcrumbsElement.innerHTML = html;
}

function updateCurrentPipeline(){
    let currentPipelineElement = document.querySelector('#currentPipeline');
    // let currentPipelineElement = document.querySelector('#pills-home');
    let html = '';

    for (let i = currentPipeline.length - 1; i >= 0; i--) {
        html +=
        `<a onclick="selectHypothesis(${i});updateCurrentPipeline();updateCurrentOperatorResults();" class="list-group-item list-group-item-action d-flex gap-3 py-3 ${(i == activeIndex ? 'active' : '')}" aria-current="true">
        <img src="${(currentPipeline[i]['action'] == 'Exploit' ? '2.png' : '3.png')}" alt="twbs" width="32" height="32" class="rounded-circle flex-shrink-0">
          <div class="d-flex gap-2 w-100 justify-content-between">
            <div>
                <h6 class="mb-0">${currentPipeline[i]['name']}</h6>
            
                <p class="mb-0 opacity-75">${currentPipeline[i]['user_count']} ${(currentPipeline[i]['user_count'] == 1 ? 'user' : 'users')}, ${currentPipeline[i]['movie_count']} ${(currentPipeline[i]['movie_count'] == 1 ? 'movie' : 'movies')}</p>
                <p class="mb-0 opacity-75">FDR: ${currentPipeline[i]['fdr']} | Power: ${currentPipeline[i]['power']} | Coverage: ${currentPipeline[i]['coverage']}</p>
                
            </div>
            <small class="opacity-50 text-nowrap">${currentPipeline[i]['size_output_set']} output ${(currentPipeline[i]['size_output_set'] == 1 ? 'region' : 'regions')}</small>
          </div>
        </a>`;
    }

    // Add the HTML to the UI
    currentPipelineElement.innerHTML = html;
    updateBreadcrumbs();
    currentPipelineToTree();
}

function updateCurrentOperatorResults(){
    let currentOperatorResultsElement = document.querySelector('#currentOperatorResults');
    let html = '';

    if (currentOperatorResults.length > 0) {
        for (let i = 0; i < currentOperatorResults[activeIndex].length; i++) {
            html +=
            `<div class="accordion-item overflow-auto">
            <h2 class="accordion-header" id="${currentOperatorResults[activeIndex][i]['heading_id']}">
                <button onclick="selectRegion(${i});updateBreadcrumbs();" class="accordion-button ${(i == 0 ? 'collapsed' : '')}" type="button" data-bs-toggle="collapse" data-bs-target="#${currentOperatorResults[activeIndex][i]['collapse_id']}" aria-expanded="${(i == 0 ? 'true' : 'false')}" aria-controls="${currentOperatorResults[activeIndex][i]['collapse_id']}">
                ${(i == 0 ? 'INPUT DATA REGION<br>' : '')}
                ${currentOperatorResults[activeIndex][i]['name']}
              </button>
            </h2>
            <div id="${currentOperatorResults[activeIndex][i]['collapse_id']}" class="accordion-collapse collapse ${(i == 0 ? 'show' : '')}" aria-labelledby="${currentOperatorResults[activeIndex][i]['heading_id']}" data-bs-parent="#currentOperatorResults">
              <div class="accordion-body overflow-auto">`;
    
            html +=
            `<!-- Ratings -->
    
                <div>
                  <canvas id="${currentOperatorResults[activeIndex][i]['heading_id']}RatingsChart"></canvas>
                </div>
                <script>
                
                  new Chart(document.getElementById("${currentOperatorResults[activeIndex][i]['heading_id']}RatingsChart"), {
                    type: 'bar',
                    data: {
                    labels: ['1', '2', '3', '4', '5'],
                    datasets: [{
                      label: 'Rating count',
                      backgroundColor: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                      data: [${currentOperatorResults[activeIndex][i]['rating_count']}],
                      borderWidth: 1
                    }]
                  },
                    options: {
                      plugins: {
                      legend: { display: false },
                      title: {
                        display: true,
                        text: 'Ratings'
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        },
                      }
                    }
                    }
                  });
                </script>
    
                <!-- Genders -->
    
                <div>
                  <canvas id="${currentOperatorResults[activeIndex][i]['heading_id']}GendersChart"></canvas>
                </div>
                
                <script>
                
                  new Chart(document.getElementById("${currentOperatorResults[activeIndex][i]['heading_id']}GendersChart"), {
                    type: 'bar',
                    data: {
                    labels: ['Male', 'Female'],
                    datasets: [{
                      label: 'User count',
                      backgroundColor: ['#1f77b4', '#ff7f0e'],
                      data: [${currentOperatorResults[activeIndex][i]['genders']}],
                      borderWidth: 1
                    }]
                  },
                    options: {
                      plugins: {
                      legend: { display: false },
                      title: {
                        display: true,
                        text: 'Genders'
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        },
                      }
                    }
                    }
                  });
                </script>
    
                <!-- Ages -->
    
                <div>
                  <canvas id="${currentOperatorResults[activeIndex][i]['heading_id']}AgesChart"></canvas>
                </div>
                
                <script>
                
                  new Chart(document.getElementById("${currentOperatorResults[activeIndex][i]['heading_id']}AgesChart"), {
                    type: 'bar',
                    data: {
                    labels: ["<18", "18-24", "25-34", "35-44", "45-49", "50-55", ">56"],
                    datasets: [{
                      label: 'User count',
                      backgroundColor: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'],
                      data: [${currentOperatorResults[activeIndex][i]['ages']}],
                      borderWidth: 1
                    }]
                  },
                    options: {
                      plugins: {
                      legend: { display: false },
                      title: {
                        display: true,
                        text: 'Ages'
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        },
                      }
                    }
                    }
                  });
                </script>
    
                <!-- Occupations -->
    
                <div>
                  <canvas id="${currentOperatorResults[activeIndex][i]['heading_id']}OccupationsChart"></canvas>
                </div>
    
                <script>
    
                  new Chart(document.getElementById("${currentOperatorResults[activeIndex][i]['heading_id']}OccupationsChart"), {
                    type: 'bar',
                    data: {
                    labels: ['Academic-educator', 'Artist', 'Clerical-admin', 'College-grad student', 'Customer service', 'Doctor-health care', 'Executive-managerial', 'Farmer', 'Homemaker', 'K-12 student', 'Lawyer', 'Programmer', 'Retired', 'Sales-marketing', 'Scientist', 'Self-employed', 'Technician-engineer', 'Tradesman-craftsman', 'Unemployed', 'Writer', 'Other'],
                    datasets: [{
                      label: 'User count',
                      backgroundColor: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4'],
                      data: [${currentOperatorResults[activeIndex][i]['occupations']}],
                      borderWidth: 1
                    }]
                  },
                    options: {
                      plugins: {
                      legend: { display: false },
                      title: {
                        display: true,
                        text: 'Occupations'
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        },
                      }
                    }
                    }
                  });
                </script>
    
                <!-- Genres -->
    
                <div>
                  <canvas id="${currentOperatorResults[activeIndex][i]['heading_id']}GenresChart"></canvas>
                </div>
    
                <script>
    
                  new Chart(document.getElementById("${currentOperatorResults[activeIndex][i]['heading_id']}GenresChart"), {
                    type: 'bar',
                    data: {
                    labels: ['Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western'],
                    datasets: [{
                      label: 'User count',
                      backgroundColor: ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'],
                      data: [${currentOperatorResults[activeIndex][i]['genres']}],
                      borderWidth: 1
                    }]
                  },
                    options: {
                      plugins: {
                      legend: { display: false },
                      title: {
                        display: true,
                        text: 'Genres'
                      },
                      scales: {
                        y: {
                          beginAtZero: true
                        },
                      }
                    }
                    }
                  });
                </script>
    
    
              `;
            
            /*
            html += `
              <div class="table-responsive">
              <table class="table">
              <thead>
                <tr>
                  <th scope="col">User ID</th>
                  <th scope="col">Movie ID</th>
                  <th scope="col">Rating</th>
                  <th scope="col">Date</th>
                  <th scope="col">Gender</th>
                  <th scope="col">Age</th>
                  <th scope="col">Occupation</th>
                  <th scope="col">Genre</th>
                  <th scope="col">Runtime Minutes</th>
                  <th scope="col">Year</th>
                </tr>
              </thead>
              <tbody>
            `;

            for (let dataIndex=0; dataIndex < currentOperatorResults[activeIndex][i]['df_data'].length; dataIndex++) {
              currentRow = currentOperatorResults[activeIndex][i]['df_data'][dataIndex];
              html += `<tr>`;
              for (let columnIndex=0; columnIndex < currentRow.length - 1; columnIndex++) {
                html += `<td>${currentRow[columnIndex]}</td>`;
              }
              html += `</tr>`;
            }

            html += `</tbody></table></div>`;
            */
           
            html += `</div></div></div>`;
        }
    }

    currentOperatorResultsElement.innerHTML = html;

    var scripts = currentOperatorResultsElement.getElementsByTagName("script");
    for (let i = 0; i < scripts.length; i++) {
        eval(scripts[i].innerText);
    }
    updateBreadcrumbs();
}

function nextStepExploration(triggeredByNext=false, agg_function=null){
    if (triggeredByNext) {
      document.getElementById("next").classList.add('disabled');
    }

    var dataset = document.getElementById("dataset").value;
    var policy = document.getElementById("policy").value;
    var selectedGroup;
    var listRegions;
    var aggregation;

    if (currentPipeline.length > 0) {
      selectedGroup = currentOperatorResults[activeIndex][selectedRegionIndex]['data_region'];
      if (selectedRegionIndex == 0) {
        listRegions = currentPipeline[activeIndex]['output_data_regions'];
      }
      aggregation = currentPipeline[activeIndex]['aggregation'];
    }

    if (agg_function != null) {
      aggregation = agg_function;
    }

    callAgent(dataset, policy, selectedGroup, listRegions, aggregation).then(data => {
        if (start || triggeredByNext) {
          console.log(data['pipeline']);
          console.log(data['operator_results']);
          currentOperatorResults.push(data['operator_results']);
          currentPipeline.push(data['pipeline']);
          activeIndex = currentPipeline.length - 1;
          selectedRegionIndex = 0;

          updateCurrentPipeline();
          updateCurrentOperatorResults();
          if (triggeredByNext) {
            document.getElementById("next").classList.remove('disabled');
          }
        }
        
    });
}

function startOrStopButton(){
    start = !start;

    if (start) {
        // Start exploration
        document.getElementById("startStop").textContent = 'Stop';
        document.getElementById("startStop").className = 'btn btn-outline-warning me-2';

        document.getElementById("next").classList.add('disabled');

        startExploration = setInterval(function(){
            nextStepExploration();
        }, 10000);
    } else {
        // Stop exploration
        document.getElementById("startStop").textContent = 'Start';
        document.getElementById("startStop").className = 'btn btn-outline-success me-2';

        document.getElementById("next").classList.remove('disabled');

        clearInterval(startExploration);
    }
    
    // document.getElementById("msg").innerHTML = "The button has been clicked.";
}

function nextButton(){
  nextStepExploration(true);
}

function nextH1(){
  nextStepExploration(true, "mean");
}

function nextH2(){
  nextStepExploration(true, "variance");
}

function nextH3(){
  nextStepExploration(true, "distribution");
}

function resetButton(){
    activeIndex = -1;
    selectedRegionIndex = 0;
    currentPipeline = [];
    currentOperatorResults = [];

    updateCurrentPipeline();
    updateCurrentOperatorResults();
}

const readFile = (file = {}, method = 'readAsText') => {
  const reader = new FileReader()
  return new Promise((resolve, reject) => {
    reader[method](file)
    reader.onload = () => {
      resolve(reader)
    }
    reader.onerror = (error) => reject(error)
  })
}

async function loadPipeline() {
  let file = document.getElementById("upload").files[0];
  const resp = await readFile(file);
  var data =  JSON.parse(resp.result);
  currentPipeline = data[0];
  currentOperatorResults = data[1];
  activeIndex = currentPipeline.length - 1;
  selectedRegionIndex = 0;
  updateCurrentPipeline();
  updateCurrentOperatorResults();
  document.getElementById('upload').value = "";

  // Generate json

  //console.log(currentPipeline);
  //console.log(currentOperatorResults);

  //console.log(tree);
  //console.log(nodeToJSON(tree));

  currentPipelineToTree();

  //currentPipelineToCSV();
}

document.getElementById("upload").onchange = function() {
  loadPipeline();
};

function downloadPipeline() {
  var a = document.createElement("a");
  // var file = new Blob(JSON.stringify({'current_pipeline': currentPipeline, 'current_operator_results': currentOperatorResults}));
  var file = new Blob([JSON.stringify([currentPipeline, currentOperatorResults])]);
  a.href = URL.createObjectURL(file);
  a.download = 'file.sheva';
  a.click();
}

function toggleAlert(){
  let alertElement = document.querySelector('#changeDataset');
  alertElement.innerHTML = `
      <div class="alert alert-primary alert-dismissible fade show" role="alert">
      The dataset has been changed
      <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
      </div>
  `;
}

function changeDataset() {
  toggleAlert();
  var selectedDataset = document.getElementById("dataset").value;
  if (selectedDataset != currentDataset) {
    if (currentPipeline.length > 0) {
      downloadPipeline();
    }
    currentDataset = selectedDataset;
    resetButton();
    //alert("New dataset selected.");
  }
}

// Get reference to button and add event listener for action "click"
document.getElementById("startStop").addEventListener("click", startOrStopButton);
document.getElementById("next").addEventListener("click", nextButton);
document.getElementById("nextH1").addEventListener("click", nextH1);
document.getElementById("nextH2").addEventListener("click", nextH2);
document.getElementById("nextH3").addEventListener("click", nextH3);
document.getElementById("reset").addEventListener("click", resetButton);
document.getElementById("dataset").addEventListener("change", changeDataset);
